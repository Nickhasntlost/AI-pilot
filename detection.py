import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading

# ---------- CONFIGURATION ----------
EAR_THRESHOLD = 0.23           # Eye aspect ratio below this = closed eye
EAR_CONSEC_FRAMES = 30         # Frames to trigger drowsy alert
MAR_THRESHOLD = 0.6            # Mouth aspect ratio above this = yawn
MAR_CONSEC_FRAMES = 15         # Frames to trigger yawn alert
ALERT_SOUND = "alarm.wav"      # Path to your alert sound

# ---------- MEDIA PIPE INITIALIZATION ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ---------- LANDMARK INDICES ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]  # For MAR

# ---------- HELPER FUNCTIONS ----------
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):
    A = euclidean_dist(eye_points[1], eye_points[5])
    B = euclidean_dist(eye_points[2], eye_points[4])
    C = euclidean_dist(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(coords):
    A = euclidean_dist(coords[MOUTH[0]], coords[MOUTH[1]])  # vertical
    B = euclidean_dist(coords[MOUTH[4]], coords[MOUTH[5]])  # vertical
    C = euclidean_dist(coords[MOUTH[2]], coords[MOUTH[3]])  # horizontal
    return (A + B) / (2.0 * C)

def play_alert():
    threading.Thread(target=lambda: playsound(ALERT_SOUND), daemon=True).start()

# ---------- MAIN LOOP ----------
cap = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

COUNTER_DROWSY = 0
COUNTER_YAWN = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # EAR calculation
            left_eye = [coords[i] for i in LEFT_EYE]
            right_eye = [coords[i] for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # MAR calculation
            mar = mouth_aspect_ratio(coords)

            # Display EAR & MAR
            cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ----- Drowsiness check -----
            if ear < EAR_THRESHOLD:
                COUNTER_DROWSY += 1
                if COUNTER_DROWSY >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSY ALERT!", (150, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    play_alert()
            else:
                COUNTER_DROWSY = 0

            # ----- Yawn check -----
            if mar > MAR_THRESHOLD:
                COUNTER_YAWN += 1
                if COUNTER_YAWN >= MAR_CONSEC_FRAMES:
                    cv2.putText(frame, "YAWNING ALERT!", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    play_alert()
            else:
                COUNTER_YAWN = 0

            # ----- Head pose (commented out) -----
            """
            # Head pose estimation code would go here
            # but is disabled for now to avoid false 'DISTRACTED ALERT's
            """

    cv2.imshow("Driver Alertness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
