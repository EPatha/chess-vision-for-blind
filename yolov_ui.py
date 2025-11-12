#!/usr/bin/env python3
"""Simple, clean YOLOv8 PyQt UI for DroidCam / VideoCapture sources.

This file provides a compact, well-indented implementation with:
- OpenCV VideoCapture primary path
- MJPEG fallback via requests when OpenCV can't open the HTTP URL
- Optional ultralytics YOLO inference (if installed)

Run with the project's venv:
    .venv/bin/python3 yolov_ui.py
"""
import sys
import time
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import requests

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class Worker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    finished = QtCore.pyqtSignal()

    def __init__(self, source: str, model_path: str = 'yolov8n.pt', device: str = 'cpu', fps: int = 8):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.device = device
        self.fps = fps
        self._stop = threading.Event()

        self.model = None
        if YOLO is not None and self.model_path:
            try:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            except Exception as e:
                print('YOLO load failed:', e)
                self.model = None

    def stop(self) -> None:
        self._stop.set()

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            return frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb)
            try:
                ann = results[0].plot()
                if isinstance(ann, np.ndarray):
                    return ann
            except Exception:
                pass
        except Exception as e:
            print('Inference error:', e)
        return frame

    def run(self) -> None:
        src = self.source
        try:
            src_i = int(src)
        except Exception:
            src_i = src

        cap = cv2.VideoCapture(src_i)
        interval = 1.0 / max(1, int(self.fps))
        last = time.time()

        if cap.isOpened():
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = self._infer(frame)
                self.frame_ready.emit(frame)

                elapsed = time.time() - last
                to_sleep = interval - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                last = time.time()

            cap.release()
            self.finished.emit()
            return

        # MJPEG fallback
        if isinstance(src, str) and src.lower().startswith('http'):
            try:
                resp = requests.get(src, stream=True, timeout=5)
                if resp.status_code != 200:
                    print('MJPEG fallback HTTP status:', resp.status_code)
                    self.finished.emit()
                    return

                buf = b''
                for chunk in resp.iter_content(chunk_size=1024):
                    if self._stop.is_set():
                        break
                    if not chunk:
                        continue
                    buf += chunk
                    a = buf.find(b'\xff\xd8')
                    b = buf.find(b'\xff\xd9')
                    if a != -1 and b != -1 and b > a:
                        jpg = buf[a:b+2]
                        buf = buf[b+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        frame = self._infer(img)
                        self.frame_ready.emit(frame)

                        elapsed = time.time() - last
                        to_sleep = interval - elapsed
                        if to_sleep > 0:
                            time.sleep(to_sleep)
                        last = time.time()

                resp.close()
            except Exception as e:
                print('MJPEG fallback failed:', e)

        print('Failed to open capture source:', src)
        self.finished.emit()


class YOLOGui(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('YOLOv8 DroidCam UI')
        self.resize(880, 600)

        self.srcInput = QtWidgets.QLineEdit('http://127.0.0.1:4747/video')
        self.modelInput = QtWidgets.QLineEdit('yolov8n.pt')
        self.deviceInput = QtWidgets.QLineEdit('cpu')
        self.fpsSpin = QtWidgets.QSpinBox(); self.fpsSpin.setRange(1, 30); self.fpsSpin.setValue(8)

        self.startBtn = QtWidgets.QPushButton('Start')
        self.stopBtn = QtWidgets.QPushButton('Stop'); self.stopBtn.setEnabled(False)

        form = QtWidgets.QFormLayout()
        form.addRow('Source', self.srcInput)
        form.addRow('Model', self.modelInput)
        form.addRow('Device', self.deviceInput)
        form.addRow('FPS', self.fpsSpin)

        top = QtWidgets.QHBoxLayout()
        top.addLayout(form)
        btns = QtWidgets.QVBoxLayout(); btns.addWidget(self.startBtn); btns.addWidget(self.stopBtn); btns.addStretch()
        top.addLayout(btns)

        self.preview = QtWidgets.QLabel(); self.preview.setFixedSize(800, 450); self.preview.setStyleSheet('background:#000')

        main = QtWidgets.QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(self.preview)

        self.thread = None
        self.worker = None

        self.startBtn.clicked.connect(self.start)
        self.stopBtn.clicked.connect(self.stop)

    @QtCore.pyqtSlot(np.ndarray)
    def on_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        qimg = QtGui.QImage(frame.data, w, h, 3*w, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview.width(), self.preview.height(), QtCore.Qt.KeepAspectRatio)
        self.preview.setPixmap(pix)

    def start(self) -> None:
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

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = YOLOGui()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""YOLOv8 PyQt UI: point at a VideoCapture source (DroidCam URL or device index)
and preview detections live.

Usage:
    .venv/bin/python3 yolov_ui.py

Controls:
 - Source: VideoCapture string (e.g. 0 or http://127.0.0.1:4747/video)
 - Model: path to yolov8 .pt file (default: yolov8n.pt)
 - Device: cpu or cuda
 - Start/Stop buttons
"""
import sys
import time
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import requests

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class Worker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    finished = QtCore.pyqtSignal()

    def __init__(self, source, model_path='yolov8n.pt', device='cpu', fps=10):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.device = device
        self.fps = fps
        self._stop = threading.Event()

        self.model = None
        if YOLO is not None and self.model_path:
            try:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            except Exception as e:
                print('YOLO load failed:', e)
                self.model = None

    def stop(self):
        self._stop.set()

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            return frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb)
            try:
                ann = results[0].plot()
                if isinstance(ann, np.ndarray):
                    return ann
            except Exception:
                pass
        except Exception as e:
            print('Inference error:', e)
        return frame

    def run(self):
        src = self.source
        try:
            src_i = int(src)
        except Exception:
            src_i = src

        cap = cv2.VideoCapture(src_i)
        interval = 1.0 / max(1, int(self.fps))
        last = time.time()

        if cap.isOpened():
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = self._infer(frame)
                self.frame_ready.emit(frame)

                elapsed = time.time() - last
                to_sleep = interval - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                last = time.time()

            cap.release()
            self.finished.emit()
            return

        # MJPEG fallback
        if isinstance(src, str) and src.lower().startswith('http'):
            try:
                resp = requests.get(src, stream=True, timeout=5)
                if resp.status_code != 200:
                    print('MJPEG fallback HTTP status:', resp.status_code)
                    self.finished.emit()
                    return

                buf = b''
                for chunk in resp.iter_content(chunk_size=1024):
                    if self._stop.is_set():
                        break
                    if not chunk:
                        continue
                    buf += chunk
                    a = buf.find(b'\xff\xd8')
                    b = buf.find(b'\xff\xd9')
                    if a != -1 and b != -1 and b > a:
                        jpg = buf[a:b+2]
                        buf = buf[b+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        frame = self._infer(img)
                        self.frame_ready.emit(frame)

                        elapsed = time.time() - last
                        to_sleep = interval - elapsed
                        if to_sleep > 0:
                            time.sleep(to_sleep)
                        last = time.time()

                resp.close()
            except Exception as e:
                print('MJPEG fallback failed:', e)

        print('Failed to open capture source:', src)
        self.finished.emit()


class YOLOGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOv8 DroidCam UI')
        self.resize(880, 600)

        self.srcInput = QtWidgets.QLineEdit('http://127.0.0.1:4747/video')
        self.modelInput = QtWidgets.QLineEdit('yolov8n.pt')
        self.deviceInput = QtWidgets.QLineEdit('cpu')
        self.fpsSpin = QtWidgets.QSpinBox(); self.fpsSpin.setRange(1, 30); self.fpsSpin.setValue(8)

        self.startBtn = QtWidgets.QPushButton('Start')
        self.stopBtn = QtWidgets.QPushButton('Stop'); self.stopBtn.setEnabled(False)

        form = QtWidgets.QFormLayout()
        form.addRow('Source', self.srcInput)
        form.addRow('Model', self.modelInput)
        form.addRow('Device', self.deviceInput)
        form.addRow('FPS', self.fpsSpin)

        top = QtWidgets.QHBoxLayout()
        top.addLayout(form)
        btns = QtWidgets.QVBoxLayout(); btns.addWidget(self.startBtn); btns.addWidget(self.stopBtn); btns.addStretch()
        top.addLayout(btns)

        self.preview = QtWidgets.QLabel(); self.preview.setFixedSize(800, 450); self.preview.setStyleSheet('background:#000')

        main = QtWidgets.QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(self.preview)

        self.thread = None
        self.worker = None

        self.startBtn.clicked.connect(self.start)
        self.stopBtn.clicked.connect(self.stop)

    @QtCore.pyqtSlot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
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
#!/usr/bin/env python3
"""YOLOv8 PyQt UI: point at a VideoCapture source (DroidCam URL or device index)
and preview detections live.

This is a cleaned, self-contained implementation that:
- opens an OpenCV VideoCapture (preferred)
- if that fails and the source is an HTTP URL, attempts an MJPEG fallback
- runs YOLO inference (if `ultralytics` is installed) and displays annotated frames

Usage:
    .venv/bin/python3 yolov_ui.py

"""
import sys
import time
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import requests

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
        if YOLO is not None and self.model_path:
            try:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            except Exception as e:
                print('YOLO load failed:', e)
                self.model = None

    def stop(self):
        self._stop.set()

    def _do_inference(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            return frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb)
            try:
                ann = results[0].plot()
                if isinstance(ann, np.ndarray):
                    return ann
            except Exception:
                pass
        except Exception as e:
            print('Inference error:', e)
        return frame

    def run(self):
        src = self.source
        try:
            src_i = int(src)
        except Exception:
            src_i = src

        cap = cv2.VideoCapture(src_i)
        interval = 1.0 / max(1, int(self.fps))
        last = time.time()

        # Primary loop: OpenCV VideoCapture
        if cap.isOpened():
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = self._do_inference(frame)
                self.frame_ready.emit(frame)

                elapsed = time.time() - last
                to_sleep = interval - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                last = time.time()

            cap.release()
            self.finished.emit()
            return

        # MJPEG fallback for HTTP endpoints
        if isinstance(src, str) and src.lower().startswith('http'):
            try:
                resp = requests.get(src, stream=True, timeout=5)
                if resp.status_code != 200:
                    print('MJPEG fallback HTTP status:', resp.status_code)
                    self.finished.emit()
                    return

                bytes_buf = b''
                for chunk in resp.iter_content(chunk_size=1024):
                    if self._stop.is_set():
                        break
                    if not chunk:
                        continue
                    bytes_buf += chunk
                    a = bytes_buf.find(b'\xff\xd8')
                    b = bytes_buf.find(b'\xff\xd9')
                    if a != -1 and b != -1 and b > a:
                        jpg = bytes_buf[a:b+2]
                        bytes_buf = bytes_buf[b+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        frame = self._do_inference(img)
                        self.frame_ready.emit(frame)

                        elapsed = time.time() - last
                        to_sleep = interval - elapsed
                        if to_sleep > 0:
                            time.sleep(to_sleep)
                        last = time.time()

                resp.close()
            except Exception as e:
                print('MJPEG fallback failed:', e)

        print('Failed to open capture source:', src)
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
import requests

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
        interval = 1.0 / max(1, int(self.fps))
        last = time.time()

        # Primary loop using OpenCV VideoCapture when available
        if cap.isOpened():
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
                        try:
                            ann = results[0].plot()
                            if isinstance(ann, np.ndarray):
                                frame = ann
                        except Exception:
                            pass
                    except Exception as e:
                        print('Inference error:', e)

                self.frame_ready.emit(frame)

                elapsed = time.time() - last
                to_sleep = interval - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                last = time.time()

            cap.release()
            self.finished.emit()
            return

        # If OpenCV couldn't open the source and it's an HTTP URL, try MJPEG fallback
        if isinstance(src, str) and src.lower().startswith('http'):
            try:
                resp = requests.get(src, stream=True, timeout=5)
                if resp.status_code != 200:
                    print('MJPEG fallback HTTP status:', resp.status_code)
                    self.finished.emit()
                    return

                bytes_buf = b''
                for chunk in resp.iter_content(chunk_size=1024):
                    if self._stop.is_set():
                        break
                    if not chunk:
                        continue
                    bytes_buf += chunk
                    a = bytes_buf.find(b'\xff\xd8')
                    b = bytes_buf.find(b'\xff\xd9')
                    if a != -1 and b != -1 and b > a:
                        jpg = bytes_buf[a:b+2]
                        bytes_buf = bytes_buf[b+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        frame = img

                        # inference
                        if self.model is not None:
                            try:
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = self.model(rgb)
                                try:
                                    ann = results[0].plot()
                                    if isinstance(ann, np.ndarray):
                                        frame = ann
                                except Exception:
                                    pass
                            except Exception as e:
                                print('Inference error:', e)

                        self.frame_ready.emit(frame)

                        elapsed = time.time() - last
                        to_sleep = interval - elapsed
                        if to_sleep > 0:
                            time.sleep(to_sleep)
                        last = time.time()

                resp.close()
            except Exception as e:
                print('MJPEG fallback failed:', e)

        print('Failed to open capture source:', src)
        self.finished.emit()
                            frame = img

                            # inference
                            if self.model is not None:
                                try:
                                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    results = self.model(rgb)
                                    try:
                                        ann = results[0].plot()
                                        if isinstance(ann, np.ndarray):
                                            frame = ann
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print('Inference error:', e)

                            self.frame_ready.emit(frame)

                            # regulate FPS roughly
                            elapsed = time.time() - last
                            to_sleep = interval - elapsed
                            if to_sleep > 0:
                                time.sleep(to_sleep)
                            last = time.time()

                    resp.close()
                except Exception as e:
                    print('MJPEG fallback failed:', e)

            print('Failed to open capture source:', src)
            self.finished.emit()

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
