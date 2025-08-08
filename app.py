import time
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response
import simpleaudio as sa  # non-blocking audio playback

# ---------- CONFIG ----------
EAR_THRESHOLD = 0.23          # Eye Aspect Ratio threshold for closed eyes
MAR_THRESHOLD = 0.60          # Mouth Aspect Ratio threshold for yawning

DROWSY_MIN_DURATION = 1.2     # Seconds EAR below threshold to trigger drowsiness alert
YAWN_MIN_DURATION = 0.7       # Seconds MAR above threshold to trigger yawn alert

SMOOTH_WINDOW = 5             # Rolling median window size for smoothing EAR/MAR

ALERT_SOUND_PATH = "static/alarm.wav"
ALERT_COOLDOWN_SEC = 3.0      # Minimum seconds between alerts

FRAME_WIDTH = 640             # Width to resize frames for processing

DEBUG_TEXT = True             # Show EAR/MAR and alert text on frame

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# ---------- FLASK APP ----------
app = Flask(__name__)

# ---------- MEDIAPIPE INITIALIZATION ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- LANDMARK INDICES ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]

# ---------- CAMERA SETUP ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

# ---------- UTILITIES ----------
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):
    A = euclidean_dist(eye_points[1], eye_points[5])
    B = euclidean_dist(eye_points[2], eye_points[4])
    C = euclidean_dist(eye_points[0], eye_points[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(coords):
    A = euclidean_dist(coords[MOUTH[0]], coords[MOUTH[1]])  # vertical
    B = euclidean_dist(coords[MOUTH[4]], coords[MOUTH[5]])  # vertical
    C = euclidean_dist(coords[MOUTH[2]], coords[MOUTH[3]])  # horizontal
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def rolling_median(deq):
    if not deq:
        return 0.0
    arr = np.array(deq, dtype=float)
    return float(np.median(arr))

# ---------- ALERT SOUND WITH COOLDOWN ----------
alert_lock = threading.Lock()
last_alert_time = 0.0
is_alert_playing = False

def _play_sound_blocking(path):
    global is_alert_playing
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception:
        pass
    finally:
        with alert_lock:
            is_alert_playing = False

def trigger_alert():
    global last_alert_time, is_alert_playing
    now = time.time()
    with alert_lock:
        if (now - last_alert_time) < ALERT_COOLDOWN_SEC:
            return
        if is_alert_playing:
            return
        is_alert_playing = True
        last_alert_time = now
    t = threading.Thread(target=_play_sound_blocking, args=(ALERT_SOUND_PATH,), daemon=True)
    t.start()

# ---------- FRAME GENERATOR ----------
def generate_frames():
    ear_below_since = None
    mar_above_since = None
    ear_deq = deque(maxlen=SMOOTH_WINDOW)
    mar_deq = deque(maxlen=SMOOTH_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        scale = FRAME_WIDTH / float(w) if w > 0 else 1.0
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame

        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        try:
            results = face_mesh.process(rgb)
        except Exception:
            results = None

        drowsy_alert = False
        yawn_alert = False
        ear_val = None
        mar_val = None

        if results and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            sh, sw = frame_small.shape[:2]
            coords = [(int(lm.x * sw), int(lm.y * sh)) for lm in face_landmarks.landmark]

            left_eye = [coords[i] for i in LEFT_EYE]
            right_eye = [coords[i] for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            ear_deq.append(ear)
            ear_val = rolling_median(ear_deq)

            mar = mouth_aspect_ratio(coords)
            mar_deq.append(mar)
            mar_val = rolling_median(mar_deq)

            now = time.time()

            # Drowsiness check
            if ear_val < EAR_THRESHOLD:
                if ear_below_since is None:
                    ear_below_since = now
                elif (now - ear_below_since) >= DROWSY_MIN_DURATION:
                    drowsy_alert = True
            else:
                ear_below_since = None

            # Yawn check
            if mar_val > MAR_THRESHOLD:
                if mar_above_since is None:
                    mar_above_since = now
                elif (now - mar_above_since) >= YAWN_MIN_DURATION:
                    yawn_alert = True
            else:
                mar_above_since = None

            # Draw info
            if DEBUG_TEXT:
                cv2.putText(frame_small, f"EAR: {ear_val:.2f}  MAR: {mar_val:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if drowsy_alert:
                    cv2.putText(frame_small, "DROWSY ALERT!", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if yawn_alert:
                    cv2.putText(frame_small, "YAWNING ALERT!", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            if drowsy_alert or yawn_alert:
                trigger_alert()

        else:
            # No face detected
            if DEBUG_TEXT:
                cv2.putText(frame_small, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            ear_below_since = None
            mar_above_since = None

        try:
            success, buffer = cv2.imencode(".jpg", frame_small)
            if not success:
                continue
            jpg_bytes = buffer.tobytes()
        except Exception:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------- CLEANUP ON EXIT ----------
def cleanup():
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except Exception:
        pass
    try:
        if face_mesh is not None:
            face_mesh.close()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

import atexit
atexit.register(cleanup)

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
