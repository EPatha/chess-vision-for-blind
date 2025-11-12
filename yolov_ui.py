#!/usr/bin/env python3
"""
Optimized YOLOv8 PyQt5 GUI Viewer
- Baca kamera lokal atau stream MJPEG (DroidCam, IP cam, dll)
- Jalankan YOLO inference jika model tersedia
- Efisien dan ringkas, cocok untuk realtime testing

Run:
    python yolov_ui.py
"""

import sys, time, threading, cv2, numpy as np, requests
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class Worker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    finished = QtCore.pyqtSignal()

    def __init__(self, source, model_path=None, device="cpu", fps=10):
        super().__init__()
        self.source = source
        self.device = device
        self.fps = max(1, int(fps))
        self._stop = threading.Event()

        self.model = None
        if YOLO and model_path:
            try:
                self.model = YOLO(model_path)
                self.model.to(device)
                print(f"[INFO] YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"[WARN] Failed to load YOLO model: {e}")

    def stop(self):
        self._stop.set()

    def _infer(self, frame):
        if not self.model:
            return frame
        try:
            res = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return res[0].plot() if isinstance(res[0].plot(), np.ndarray) else frame
        except Exception:
            return frame

    def _open_mjpeg(self):
        try:
            resp = requests.get(self.source, stream=True, timeout=5)
            if resp.status_code != 200:
                print(f"[ERROR] HTTP {resp.status_code}")
                return
            buf, interval, last = b"", 1 / self.fps, time.time()
            for chunk in resp.iter_content(chunk_size=1024):
                if self._stop.is_set():
                    break
                buf += chunk
                a, b = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                if a != -1 and b != -1 and b > a:
                    jpg = buf[a:b+2]; buf = buf[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is None: continue
                    frame = self._infer(img)
                    self.frame_ready.emit(frame)
                    sleep = interval - (time.time() - last)
                    if sleep > 0: time.sleep(sleep)
                    last = time.time()
            resp.close()
        except Exception as e:
            print("[WARN] MJPEG fallback failed:", e)

    def run(self):
        try:
            src = int(self.source)
        except ValueError:
            src = self.source

        cap = cv2.VideoCapture(src)
        interval, last = 1 / self.fps, time.time()

        if cap.isOpened():
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                frame = self._infer(frame)
                self.frame_ready.emit(frame)
                sleep = interval - (time.time() - last)
                if sleep > 0: time.sleep(sleep)
                last = time.time()
            cap.release()
        else:
            self._open_mjpeg()
        self.finished.emit()


class YOLOGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Live Viewer")
        self.resize(880, 600)

        # Inputs
        self.src = QtWidgets.QLineEdit("0")
        self.model = QtWidgets.QLineEdit("yolov8n.pt")
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.fps = QtWidgets.QSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(10)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)

        form = QtWidgets.QFormLayout()
        form.addRow("Source:", self.src)
        form.addRow("Model:", self.model)
        form.addRow("Device:", self.device)
        form.addRow("FPS:", self.fps)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addLayout(form)
        btns = QtWidgets.QVBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch()
        control_layout.addLayout(btns)

        # Video display
        self.preview = QtWidgets.QLabel()
        self.preview.setFixedSize(800, 450)
        self.preview.setStyleSheet("background:black")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(control_layout)
        layout.addWidget(self.preview)

        self.thread = None
        self.worker = None

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def _on_frame(self, frame):
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
        self.worker = Worker(src, model_path=model, device=device, fps=fps)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
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
