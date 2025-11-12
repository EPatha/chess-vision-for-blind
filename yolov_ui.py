#!/usr/bin/env python3
"""Simple PyQt5 UI to point at a VideoCapture source and run YOLOv8 preview.

This UI lets you enter a DroidCam URL (or local device index), start a background
worker that captures frames and runs YOLO inference, and shows the annotated
preview in the window.

Usage:
    python3 yolov_ui.py

Controls:
 - Source: VideoCapture string (e.g. 0 or http://127.0.0.1:4747/video)
 - Model: path to yolov8 .pt file (default: yolov8n.pt)
 - Device: cpu or cuda
 - Start / Stop buttons
"""
import sys
import time
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class Worker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    finished = QtCore.pyqtSignal()

    def __init__(self, source, model_path, device, fps=15):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.device = device
        self.fps = fps
        self._stop = threading.Event()

        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            except Exception as e:
                print('YOLO load failed:', e)
                self.model = None

    def stop(self):
        self._stop.set()

    def run(self):
        src = self.source
        try:
            src_i = int(src)
        except Exception:
            src_i = src

        cap = cv2.VideoCapture(src_i)
        if not cap.isOpened():
            print('Failed to open capture source:', src)
            self.finished.emit()
            return

        interval = 1.0 / max(1, int(self.fps))
        last = time.time()

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # inference
            if self.model is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.model(rgb)
                    # prefer results[0].plot() if available
                    try:
                        ann = results[0].plot()
                        if isinstance(ann, np.ndarray):
                            frame = ann
                    except Exception:
                        # fallback: do nothing
                        pass
                except Exception as e:
                    print('Inference error:', e)

            # emit numpy BGR frame
            self.frame_ready.emit(frame)

            # sleep to regulate FPS
            elapsed = time.time() - last
            to_sleep = interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            last = time.time()

        cap.release()
        self.finished.emit()


class YOLOGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOv8 DroidCam UI')
        self.setGeometry(200, 100, 900, 600)

        self.srcInput = QtWidgets.QLineEdit('http://127.0.0.1:4747/video')
        self.modelInput = QtWidgets.QLineEdit('yolov8n.pt')
        self.deviceInput = QtWidgets.QLineEdit('cpu')
        self.fpsSpin = QtWidgets.QSpinBox(); self.fpsSpin.setRange(1, 60); self.fpsSpin.setValue(10)

        self.startBtn = QtWidgets.QPushButton('Start')
        self.stopBtn = QtWidgets.QPushButton('Stop'); self.stopBtn.setEnabled(False)

        form = QtWidgets.QFormLayout()
        form.addRow('Source', self.srcInput)
        form.addRow('Model', self.modelInput)
        form.addRow('Device', self.deviceInput)
        form.addRow('FPS', self.fpsSpin)

        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addLayout(form)
        btnLayout = QtWidgets.QVBoxLayout(); btnLayout.addWidget(self.startBtn); btnLayout.addWidget(self.stopBtn)
        topLayout.addLayout(btnLayout)

        self.preview = QtWidgets.QLabel(); self.preview.setFixedSize(800, 450); self.preview.setStyleSheet('background:#000')

        main = QtWidgets.QVBoxLayout(self)
        main.addLayout(topLayout)
        main.addWidget(self.preview)

        self.thread = None
        self.worker = None

        self.startBtn.clicked.connect(self.start)
        self.stopBtn.clicked.connect(self.stop)

    @QtCore.pyqtSlot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        # convert to QImage
        qimg = QtGui.QImage(frame.data, w, h, 3*w, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview.width(), self.preview.height(), QtCore.Qt.KeepAspectRatio)
        self.preview.setPixmap(pix)

    def start(self):
        src = self.srcInput.text().strip()
        model = self.modelInput.text().strip()
        device = self.deviceInput.text().strip()
        fps = int(self.fpsSpin.value())

        self.thread = QtCore.QThread()
        self.worker = Worker(src, model, device, fps=fps)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)

    def stop(self):
        if self.worker:
            self.worker.stop()
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = YOLOGui()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
