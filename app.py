"""Flask web UI to read a stream (DroidCam) and run YOLO detection with interactive 4-point calibration.

Features:
- /           -> web UI (input stream URL, start/stop)
- /video_feed -> MJPEG stream of annotated frames
- /snapshot   -> single JPEG snapshot (used for click calibration)
- /set_corners -> POST JSON with 4 points (TL,TR,BR,BL) to set board corners

Usage (dev):
    source: http://192.168.0.101:4747/video  (DroidCam HTTP stream)
    python3 app.py

Open http://localhost:5000 in your browser, masukkan URL stream, Start, lalu Kalibrasi untuk klik 4 sudut.
"""

from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import threading
import time
import io
import cv2
import numpy as np
import requests

# Import YOLO with compatibility for different ultralytics layouts
try:
    from ultralytics import YOLO
except Exception:
    try:
        from ultralytics.yolo import YOLO
    except Exception as e:
        print("Failed to import YOLO from ultralytics. Install with: pip install ultralytics")
        raise

from chess_board_utils import compute_perspective_transform, pixel_to_square

app = Flask(__name__)

# Global simple state (not persistent) — ok for local single-user use
state = {
    'source': None,
    'model_path': 'yolov8n.pt',
    'device': 'cpu',
    'model': None,
    'corners': None,  # list of 4 (x,y) tuples: TL,TR,BR,BL
    'M': None,
    'board_pixels': 800,
}


def get_model():
    if state['model'] is None:
        print(f"Loading model {state['model_path']} on {state['device']}")
        state['model'] = YOLO(state['model_path'])
        try:
            state['model'].to(state['device'])
        except Exception:
            pass
    return state['model']


def gen_frames(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    model = get_model()

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(source)
            continue

        # inference
        try:
            results = model(frame)
            # render annotated frame if available
            annotated = None
            try:
                annotated = results.render()[0]
            except Exception:
                try:
                    annotated = results[0].plot()
                except Exception:
                    annotated = frame

            # if corners set, map detected boxes/centers to squares and draw labels
            if state['M'] is not None:
                # collect box centers
                squares = []
                r = results[0]
                # ultralytics Results: r.boxes.xyxy, r.boxes.cls, r.boxes.conf
                try:
                    boxes = r.boxes.xyxy.cpu().numpy()
                except Exception:
                    try:
                        boxes = np.array(r.boxes.xyxy)
                    except Exception:
                        boxes = []

                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = b[:4]
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)
                    sq = pixel_to_square((cx, cy), state['M'], board_pixels=state['board_pixels'])
                    squares.append(sq)

                # draw squares list on annotated image
                cv2.putText(annotated, 'Detected squares: ' + ','.join(squares[:8]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            print('Inference error:', e)
            annotated = frame

        # encode to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


# Fallback static route: if Flask for some reason doesn't serve files from the default
# static folder, this route will explicitly serve files from ./static.
@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('static', filename)
    except Exception:
        return ("", 404)


@app.route('/set_source', methods=['POST'])
def set_source():
    data = request.json
    source = data.get('source')
    model_path = data.get('model', 'yolov8n.pt')
    device = data.get('device', 'cpu')
    state['source'] = source
    state['model_path'] = model_path
    state['device'] = device
    # reset model so it reloads with new model_path/device
    state['model'] = None
    return jsonify({'ok': True})


@app.route('/set_corners', methods=['POST'])
def set_corners():
    data = request.json
    pts = data.get('points')
    if not pts or len(pts) != 4:
        return jsonify({'ok': False, 'error': 'Provide 4 points in TL,TR,BR,BL order'}), 400
    # pts expected as list of {x:.., y:..}
    src_pts = [(float(p['x']), float(p['y'])) for p in pts]
    M, Minv, bp = compute_perspective_transform(src_pts, board_pixels=800)
    state['corners'] = src_pts
    state['M'] = M
    state['board_pixels'] = bp
    return jsonify({'ok': True})


@app.route('/snapshot')
def snapshot():
    source = request.args.get('source') or state.get('source')
    if not source:
        return 'No source provided', 400
    # Try OpenCV capture first
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                ret, buffer = cv2.imencode('.jpg', frame)
                return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        app.logger.debug('OpenCV snapshot failed: %s', e)

    # Fallback: try HTTP GET for a single-frame endpoint (DroidCam often exposes /shot.jpg or similar)
    candidates = [source, source.rstrip('/') + '/shot.jpg', source.rstrip('/') + '/image.jpg', source.rstrip('/') + '/photo.jpg']
    for url in candidates:
        try:
            resp = requests.get(url, timeout=3)
            ct = resp.headers.get('Content-Type', '')
            if resp.status_code == 200 and ('image' in ct or resp.content[:4] == b'\xff\xd8\xff\xe0'):
                return Response(resp.content, mimetype='image/jpeg')
        except Exception as e:
            app.logger.debug('HTTP snapshot try failed (%s): %s', url, e)

    return 'Cannot open source or retrieve snapshot', 500


@app.route('/video_feed')
def video_feed():
    source = request.args.get('source') or state.get('source')
    if not source:
        return 'No source', 400
    return Response(gen_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
