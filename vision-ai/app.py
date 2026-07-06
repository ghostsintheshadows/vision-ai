"""
Vision AI — Flask backend
Serves the landing page + three detection pages.
MJPEG streams webcam frames processed in real-time.
"""

import io
import os
import shutil
import pathlib
import threading
import time

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, Response, jsonify, request,
    send_from_directory, render_template_string
)

# ── Optional Tesseract ────────────────────────────────────────────────────────
try:
    from pytesseract import pytesseract, Output as TessOutput
    _tess = shutil.which("tesseract") or ""
    if _tess:
        pytesseract.tesseract_cmd = _tess
    TESSERACT_OK = bool(_tess)
except ImportError:
    TESSERACT_OK = False

app = Flask(__name__, static_folder="static")

# ── Global state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "color_name": "Green",
    "color_bgr":  [0, 255, 0],
    "text_image_path": "",
}

# Haar cascade
_cascade_xml = (
    pathlib.Path(cv2.__file__).parent.absolute()
    / "data/haarcascade_frontalface_default.xml"
)
_clf = cv2.CascadeClassifier(str(_cascade_xml))

# Camera (shared, lazy)
_cap: cv2.VideoCapture | None = None
_cap_lock = threading.Lock()


def _get_cap() -> cv2.VideoCapture | None:
    global _cap
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            _cap = cv2.VideoCapture(0)
        return _cap if _cap.isOpened() else None


def _release_cap() -> None:
    global _cap
    with _cap_lock:
        if _cap:
            _cap.release()
            _cap = None


# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "Green":  [0, 255, 0],
    "Red":    [0, 0, 255],
    "Blue":   [255, 0, 0],
    "Yellow": [0, 255, 255],
    "Orange": [0, 165, 255],
    "Purple": [128, 0, 128],
    "Cyan":   [255, 255, 0],
    "White":  [255, 255, 255],
}

COLOUR_HEX = {
    "Green": "#22c55e", "Red": "#ef4444", "Blue": "#3b82f6",
    "Yellow": "#eab308", "Orange": "#f97316", "Purple": "#a855f7",
    "Cyan": "#06b6d4", "White": "#f1f5f9",
}


def _color_limits(bgr):
    pixel = np.uint8([[bgr]])
    hsv   = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    hue   = int(hsv[0][0][0])
    lower = np.array([max(hue-15, 0),   80,  80], dtype=np.uint8)
    upper = np.array([min(hue+15, 179), 255, 255], dtype=np.uint8)
    return lower, upper


# ── Frame processors ──────────────────────────────────────────────────────────
def _process_color(frame):
    with _lock:
        bgr  = list(_state["color_bgr"])
        name = _state["color_name"]

    lower, upper = _color_limits(bgr)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    bbox = Image.fromarray(mask).getbbox()
    if bbox:
        x1, y1, x2, y2 = bbox
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(bgr), -1)
        frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(bgr), 3)
        lbl = f"{name}  [{x2-x1}x{y2-y1}px]"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+8, y1), tuple(bgr), -1)
        cv2.putText(frame, lbl, (x1+4, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


_face_params = {"scale_factor": 1.1, "min_neighbours": 5}
_face_count  = {"count": 0}


def _process_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _clf.detectMultiScale(
        gray,
        scaleFactor  = _face_params.get("scale_factor", 1.1),
        minNeighbors = _face_params.get("min_neighbours", 5),
        minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
    _face_count["count"] = len(faces)

    for i, (x, y, w, h) in enumerate(faces):
        c = (0, 210, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
        t = 18
        for cx, cy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx+dx*t, cy), c, 3)
            cv2.line(frame, (cx, cy), (cx, cy+dy*t), c, 3)
        lbl = f"Face {i+1}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y-th-12), (x+tw+8, y), c, -1)
        cv2.putText(frame, lbl, (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 15, 15), 2)

    cv2.putText(frame, f"Faces: {len(faces)}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 210, 255) if len(faces) else (130, 130, 130), 2)
    return frame


_ocr_cache: dict = {}

def _get_text_frame():
    with _lock:
        path = _state.get("text_image_path", "")
    if not path or not pathlib.Path(path).exists():
        return None
    if path in _ocr_cache:
        return _ocr_cache[path]

    img = cv2.imread(path)
    if img is None:
        return None

    if TESSERACT_OK:
        data = pytesseract.image_to_data(img, output_type=TessOutput.DICT)
        out  = img.copy()
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            try:
                conf = int(data["conf"][i])
            except:
                conf = 0
            if conf < 40:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 200, 80), 2)
            cv2.putText(out, word, (x, max(y-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 60, 220), 2)
        _ocr_cache[path] = out
    else:
        cv2.putText(img, "Tesseract not installed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        _ocr_cache[path] = img

    return _ocr_cache[path]


# ── Placeholder frame ─────────────────────────────────────────────────────────
def _placeholder(msg="No camera signal"):
    h, w = 480, 640
    img = np.full((h, w, 3), (14, 15, 22), dtype=np.uint8)
    for x in range(0, w, 64):
        cv2.line(img, (x, 0), (x, h), (28, 30, 45), 1)
    for y in range(0, h, 64):
        cv2.line(img, (0, y), (w, y), (28, 30, 45), 1)
    cv2.ellipse(img, (w//2, h//2-20), (55, 27), 0, 0, 360, (55, 55, 80), 2)
    cv2.circle(img, (w//2, h//2-20), 12, (75, 75, 110), -1)
    cv2.circle(img, (w//2, h//2-20),  5, (120, 120, 160), -1)
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(img, msg, (w//2-tw//2, h//2+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 130), 1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _frame_to_jpg(frame, q=82):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def _mjpeg(gen):
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


def _wrap(jpg):
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"


# ── MJPEG generators ──────────────────────────────────────────────────────────
def _gen_color():
    while True:
        cap = _get_cap()
        if not cap:
            yield _wrap(_placeholder("No camera — check connection"))
            time.sleep(0.5)
            continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        try:
            frame = _process_color(frame)
        except Exception as e:
            cv2.putText(frame, str(e), (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,220), 2)
        yield _wrap(_frame_to_jpg(frame))


def _gen_face():
    while True:
        cap = _get_cap()
        if not cap:
            yield _wrap(_placeholder("No camera — check connection"))
            time.sleep(0.5)
            continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        try:
            frame = _process_face(frame)
        except Exception as e:
            cv2.putText(frame, str(e), (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,220), 2)
        yield _wrap(_frame_to_jpg(frame))


def _gen_text():
    while True:
        img = _get_text_frame()
        if img is None:
            yield _wrap(_placeholder("Upload an image to analyse"))
            time.sleep(0.2)
            continue
        yield _wrap(_frame_to_jpg(img, q=88))
        time.sleep(0.2)   # static image — no need to hammer CPU


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/color")
def page_color():
    return send_from_directory("static", "color.html")

@app.route("/face")
def page_face():
    return send_from_directory("static", "face.html")

@app.route("/text")
def page_text():
    return send_from_directory("static", "text.html")

@app.route("/static/images/<path:fn>")
def serve_image(fn):
    return send_from_directory("static/images", fn)

# MJPEG streams
@app.route("/stream/color")
def stream_color():
    return _mjpeg(_gen_color())

@app.route("/stream/face")
def stream_face():
    return _mjpeg(_gen_face())

@app.route("/stream/text")
def stream_text():
    return _mjpeg(_gen_text())

# API
@app.route("/api/color", methods=["POST"])
def api_color():
    name = request.get_json(force=True).get("name", "Green")
    if name not in COLOURS:
        return jsonify({"error": "unknown colour"}), 400
    with _lock:
        _state["color_bgr"]  = COLOURS[name]
        _state["color_name"] = name
    return jsonify({"ok": True, "name": name})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f   = request.files["file"]
    up  = pathlib.Path("uploads"); up.mkdir(exist_ok=True)
    dst = up / f.filename
    f.save(str(dst))
    _ocr_cache.pop(str(dst.resolve()), None)
    with _lock:
        _state["text_image_path"] = str(dst.resolve())
    return jsonify({"ok": True, "name": f.filename})

@app.route("/api/colours")
def api_colours():
    return jsonify({k: COLOUR_HEX[k] for k in COLOURS})

@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "tesseract": TESSERACT_OK,
            "color":     _state["color_name"],
            "has_image": bool(_state["text_image_path"]),
        })


# ── Face params & count ───────────────────────────────────────────────────────
@app.route("/api/face_params", methods=["POST"])
def api_face_params():
    d = request.get_json(force=True)
    _face_params["scale_factor"]   = float(d.get("scale_factor", 1.1))
    _face_params["min_neighbours"] = int(d.get("min_neighbours", 5))
    return jsonify({"ok": True})

@app.route("/api/face_count")
def api_face_count():
    return jsonify(_face_count)


# ── OCR result ────────────────────────────────────────────────────────────────
@app.route("/api/ocr_result")
def api_ocr_result():
    with _lock:
        path = _state.get("text_image_path", "")
    if not path or not pathlib.Path(path).exists() or not TESSERACT_OK:
        return jsonify({"words": [], "avg_conf": 0})

    img = cv2.imread(path)
    if img is None:
        return jsonify({"words": [], "avg_conf": 0})

    data     = pytesseract.image_to_data(img, output_type=TessOutput.DICT)
    words    = []
    confs    = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        try:
            conf = int(data["conf"][i])
        except:
            conf = 0
        if conf < 40:
            continue
        words.append(word)
        confs.append(conf)

    avg = sum(confs) / len(confs) if confs else 0
    return jsonify({"words": words, "avg_conf": avg})


if __name__ == "__main__":
    print("\n  Vision AI  →  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
