#!/usr/bin/env python3
"""YOLOv8 + OpenCV demo that reads from a VideoCapture source (e.g. DroidCam HTTP URL).

Usage:
    python3 yolov8_opencv_demo.py --source "http://127.0.0.1:4747/video" --model yolov8n.pt --device cpu

This script opens an OpenCV window and runs YOLO inference on each frame. Press 'q' to quit.
"""
import argparse
import time
import sys

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def draw_boxes_on_frame(results, frame):
    """Draw detections from an Ultralytics results object onto the BGR frame.

    If the results object supports `plot()`, prefer that; otherwise fall back to
    using available boxes data.
    """
    try:
        # results[0].plot() returns an annotated image (numpy) in many ultralytics versions
        annotated = results[0].plot()
        if isinstance(annotated, np.ndarray):
            return annotated
    except Exception:
        pass

    # Fallback: iterate boxes
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        # boxes.xyxyn, boxes.conf, boxes.cls not guaranteed; adapt
        try:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception:
            pass
    return frame


def run(source, model_path='yolov8n.pt', device='cpu'):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Failed to open source: {source}')
        return

    if YOLO is None:
        print('ultralytics YOLO package not available; running preview only.')
        model = None
    else:
        try:
            model = YOLO(model_path) if model_path else YOLO()
            model.to(device)
        except Exception as e:
            print('Failed to load YOLO model:', e)
            model = None

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame read failed, retrying...')
            time.sleep(0.2)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if model is not None:
            try:
                results = model(rgb)
                frame = draw_boxes_on_frame(results, frame)
            except Exception as e:
                # don't crash the loop on model errors
                print('Inference error:', e)

        now = time.time()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow('YOLOv8 Preview', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', '-s', default=0, help='OpenCV VideoCapture source (int device or URL)')
    p.add_argument('--model', '-m', default='yolov8n.pt', help='Path to YOLOv8 model (yolov8n.pt)')
    p.add_argument('--device', '-d', default='cpu', help='Device for model (cpu or cuda)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # attempt int cast for local camera
    try:
        src = int(args.source)
    except Exception:
        src = args.source
    run(src, model_path=args.model, device=args.device)
