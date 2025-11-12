#!/usr/bin/env python3
"""Stream dari IP camera dan jalankan deteksi YOLO (ultralytics) per-frame.

Contoh:
    python3 stream_yolo.py --source "http://192.168.0.101:4747/video" --model yolov8n.pt

Catatan:
- Pastikan `ultralytics` dan `opencv-python` terinstall di environment Anda.
"""

import argparse
import time
import sys

import cv2

# Import YOLO with compatibility for different ultralytics package layouts
try:
    from ultralytics import YOLO
except Exception:
    try:
        # some versions expose YOLO in ultralytics.yolo
        from ultralytics.yolo import YOLO
    except Exception as e:
        print("Paket 'ultralytics' tidak ditemukan atau import YOLO gagal. Install with: pip install ultralytics")
        raise


def open_capture(source: str, width: int = None, height: int = None):
    cap = cv2.VideoCapture(source)
    # optional set resolution
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='http://192.168.0.101:4747/video', help='URL stream (http/rtsp) atau device index')
    p.add_argument('--model', default='yolov8n.pt', help='YOLO model path or name (ultralytics will auto-download if needed)')
    p.add_argument('--device', default='cpu', help='Device for inference, e.g. cpu or cuda')
    p.add_argument('--out', default=None, help='Optional output video file to save annotated stream (mp4)')
    p.add_argument('--width', type=int, default=None, help='Optional request frame width')
    p.add_argument('--height', type=int, default=None, help='Optional request frame height')
    args = p.parse_args()

    print(f"Loading model '{args.model}' on device {args.device}...")
    model = YOLO(args.model)
    # set device if supported by ultralytics
    try:
        model.to(args.device)
    except Exception:
        pass

    cap = open_capture(args.source, args.width, args.height)
    if not cap.isOpened():
        print(f"Gagal membuka source: {args.source}")
        sys.exit(1)

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        # try to get frame size from capture
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    print("Tekan 'q' pada window untuk keluar.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # stream mungkin terputus — coba reconnect beberapa kali
                print("Frame tidak didapat. Mencoba reconnect dalam 1 detik...")
                time.sleep(1.0)
                cap.release()
                cap = open_capture(args.source, args.width, args.height)
                continue

            # inference (ultralytics) — per-frame
            results = model(frame)
            # results.render() menggambar anotasi pada salinan dan mengembalikan list of images
            try:
                annotated = results.render()[0]
            except Exception:
                # alternatif: gunakan plot() pada result object
                try:
                    annotated = results[0].plot()
                except Exception:
                    # fallback: tampilkan frame asli
                    annotated = frame

            cv2.imshow("Chess Detection", annotated)

            if writer is not None:
                # ensure writer size matches annotated frame
                writer.write(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if annotated.ndim == 3 else annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
