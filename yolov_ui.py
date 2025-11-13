#!/usr/bin/env python3
"""
YOLOv8 PyQt5 Viewer (FFMPEG-first + MJPEG fallback)
- Try OpenCV VideoCapture (with CAP_FFMPEG for URLs) first.
- Fall back to a manual MJPEG parser using requests when VideoCapture fails
  or when "Force manual MJPEG" is checked.
- Optional ultralytics YOLO inference if `ultralytics` is installed and model
  path provided.
"""
import sys
import time
import threading
from typing import Optional

import cv2
import numpy as np
import requests
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class CaptureResult:
    def __init__(self, frame: np.ndarray, ts: float):
        self.frame = frame
        self.ts = ts


class Worker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(object)  # emits CaptureResult
    finished = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(str)

    def __init__(
        self,
        source: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
        fps: int = 10,
        manual_mjpeg: bool = False,
    ):
        super().__init__()
        self.source = source
        self.device = device
        self.fps = max(1, int(fps))
        self._stop = threading.Event()
        self.manual_mjpeg = bool(manual_mjpeg)

        self.model = None
        if YOLO and model_path:
            try:
                self.model = YOLO(model_path)
                self.status.emit(f"YOLO model loaded: {model_path}")
            except Exception as e:
                print("[WARN] Failed to load YOLO model:", e)

    def stop(self):
        self._stop.set()

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        if not self.model:
            return frame
        try:
            res = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                annotated = res[0].plot()
                if isinstance(annotated, np.ndarray):
                    return annotated
            except Exception:
                pass
            return frame
        except Exception:
            return frame

    def _open_mjpeg(self):
        """Manual MJPEG reader using requests; emits frames."""
        print(f"[DEBUG] Attempting manual MJPEG from: {self.source}")
        try:
            resp = requests.get(self.source, stream=True, timeout=10)
            print(f"[DEBUG] HTTP Status: {resp.status_code}")
            print(f"[DEBUG] Headers: {dict(resp.headers)}")
            resp.raise_for_status()
            
            buf = b""
            interval = 1.0 / self.fps
            last = time.time()
            frame_count = 0
            
            for chunk in resp.iter_content(chunk_size=4096):
                if self._stop.is_set():
                    print("[DEBUG] Stop signal received, exiting MJPEG reader")
                    break
                if not chunk:
                    continue
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg = buf[start : end + 2]
                    buf = buf[end + 2 :]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        print("[DEBUG] Failed to decode JPEG frame")
                        continue
                    frame_count += 1
                    if frame_count <= 3:
                        print(f"[DEBUG] Frame {frame_count} decoded successfully: {img.shape}")
                    img = self._infer(img)
                    self.frame_ready.emit(CaptureResult(img, time.time()))
                    sleep = interval - (time.time() - last)
                    if sleep > 0:
                        time.sleep(sleep)
                    last = time.time()
            try:
                resp.close()
            except Exception:
                pass
            print(f"[DEBUG] MJPEG reader finished. Total frames: {frame_count}")
        except Exception as e:
            print(f"[ERROR] MJPEG fallback failed: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        # parse source as int when possible
        try:
            src = int(self.source)
        except Exception:
            src = self.source

        # forced manual MJPEG?
        if self.manual_mjpeg:
            self.status.emit("Using manual MJPEG (forced)")
            self._open_mjpeg()
            self.finished.emit()
            return

        # try VideoCapture (prefer FFMPEG backend for URLs)
        cap = None
        try:
            if isinstance(src, int):
                cap = cv2.VideoCapture(src)
            else:
                # attempt FFMPEG backend first (may be a no-op if OpenCV build lacks it)
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    # fallback to default backend
                    cap.release()
                    cap = cv2.VideoCapture(src)
        except Exception:
            cap = None

        interval = 1.0 / self.fps
        last = time.time()

        if cap is not None and cap.isOpened():
            self.status.emit("VideoCapture opened (FFMPEG preferred)")
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                frame = self._infer(frame)
                self.frame_ready.emit(CaptureResult(frame, time.time()))
                sleep = interval - (time.time() - last)
                if sleep > 0:
                    time.sleep(sleep)
                last = time.time()
            try:
                cap.release()
            except Exception:
                pass
        else:
            self.status.emit("VideoCapture failed; falling back to manual MJPEG")
            self._open_mjpeg()

        self.finished.emit()


class YOLOGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Live Viewer")
        self.resize(900, 640)

        # Inputs
        self.src = QtWidgets.QLineEdit("0")  # Use built-in webcam for testing (change to 0, 1, or DroidCam URL)
        self.model = QtWidgets.QLineEdit("runs/chess_detect/train/weights/best.pt")  # Chess detection model
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(10)
        self.force_mjpeg = QtWidgets.QCheckBox("Force manual MJPEG (debug)")

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        form = QtWidgets.QFormLayout()
        form.addRow("Source:", self.src)
        form.addRow("Model:", self.model)
        form.addRow("Device:", self.device)
        form.addRow("FPS:", self.fps)
        form.addRow("Manual MJPEG:", self.force_mjpeg)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addLayout(form)
        btns = QtWidgets.QVBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch()
        control_layout.addLayout(btns)

        # Video display
        self.preview = QtWidgets.QLabel()
        self.preview.setFixedSize(860, 480)
        self.preview.setStyleSheet("background:black")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(control_layout)
        layout.addWidget(self.preview)

        # status bar
        self.status = QtWidgets.QStatusBar()
        layout.addWidget(self.status)

        self.thread = None
        self.worker = None

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def _on_frame(self, capres: CaptureResult):
        frame = capres.frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        qimg = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.preview.width(), self.preview.height(), QtCore.Qt.KeepAspectRatio
        )
        self.preview.setPixmap(pix)

    def start(self):
        src = self.src.text().strip()
        model = self.model.text().strip() or None
        device = self.device.currentText()
        fps = int(self.fps.value())

        self.thread = QtCore.QThread()
        force_manual = bool(self.force_mjpeg.isChecked())
        self.worker = Worker(src, model_path=model, device=device, fps=fps, manual_mjpeg=force_manual)
        self.worker.moveToThread(self.thread)

        # hooks / connections
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status.showMessage)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)


    def stop(self):
        if self.worker:
            self.worker.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = YOLOGui()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
