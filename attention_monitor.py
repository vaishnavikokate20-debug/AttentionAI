import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from threading import Thread, Lock
from app import app, monitor_data, data_lock, focus_history
from flask import Response
# Shared webcam frame for the Flask dashboard
from camera_stream import latest_frame, frame_lock
def run_flask():
    print("Starting Flask server...")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )

# Start Flask in the background
flask_thread = Thread(target=run_flask, daemon=True)
flask_thread.start()
print("Flask startup thread launched.")
# ==========================================
# 1. INITIALIZE MEDIAPIPE
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# 2. INITIALIZE WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# 3. EYE TRACKING SETTINGS
# ==========================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20
MIN_BLINK_DURATION = 0.05
MAX_BLINK_DURATION = 0.50
PROLONGED_CLOSURE_THRESHOLD = 2.0

blink_count = 0
eyes_closed = False
closed_start_time = None

# Count prolonged eye-closure alerts
drowsy_count = 0
drowsy_alert_active = False
# Screen-facing statistics
facing_screen_time = 0.0
looking_away_time = 0.0
last_frame_time = time.time()
focus_score = 0.0

# ==========================================
# 4. HEAD POSE SETTINGS
# ==========================================

LANDMARKS = [1, 152, 33, 263, 61, 291]

FACE_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

angle_history = deque(maxlen=10)

neutral_pitch = None
neutral_yaw = None

# ==========================================
# 5. EYE TRACKING FUNCTIONS
# ==========================================

def calculate_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def calculate_ear(eye_points):
    vertical1 = calculate_distance(
        eye_points[1], eye_points[5]
    )

    vertical2 = calculate_distance(
        eye_points[2], eye_points[4]
    )

    horizontal = calculate_distance(
        eye_points[0], eye_points[3]
    )

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2 * horizontal)


# ==========================================
# 6. CAMERA SETTINGS FOR HEAD POSE
# ==========================================

if not cap.isOpened():
    print("Error: Could not access webcam.")

else:
    print("AttentionAI started!")
    print("Look straight at the camera during calibration.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read webcam frame.")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)
       
        height, width, _ = frame.shape

        # Convert image to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Detect facial landmarks
        results = face_mesh.process(rgb_frame)

        eye_status = "Face not detected"
        direction = "Face not detected"
        ear_value = 0.0
        yaw_change = 0.0
        pitch_change = 0.0

        # ======================================
        # 7. PROCESS FACIAL LANDMARKS
        # ======================================

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            # ----------------------------------
            # A. EYE-BLINK DETECTION
            # ----------------------------------

            left_points = []
            right_points = []

            for index in LEFT_EYE:
                landmark = face_landmarks.landmark[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                left_points.append((x, y))

                cv2.circle(
                    frame, (x, y), 2, (0, 255, 0), -1
                )

            for index in RIGHT_EYE:
                landmark = face_landmarks.landmark[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                right_points.append((x, y))

                cv2.circle(
                    frame, (x, y), 2, (0, 255, 0), -1
                )

            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)

            ear_value = (left_ear + right_ear) / 2

            current_time = time.time()

            if ear_value < EAR_THRESHOLD:

                eye_status = "Eyes Closed"

                if not eyes_closed:
                    eyes_closed = True
                    closed_start_time = current_time
                closure_duration = current_time - closed_start_time

                if closure_duration >= PROLONGED_CLOSURE_THRESHOLD:
                    eye_status = "WARNING: Prolonged Eye Closure!"

                    if not drowsy_alert_active:
                       drowsy_count += 1
                       drowsy_alert_active = True

            else:

                eye_status = "Eyes Open"

                if eyes_closed:

                   duration = current_time - closed_start_time

                   if MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                    blink_count += 1

                   eyes_closed = False
                   closed_start_time = None
                   drowsy_alert_active = False 
            # ----------------------------------
            # B. HEAD-POSE DETECTION
            # ----------------------------------

            image_points = []

            for index in LANDMARKS:
                landmark = face_landmarks.landmark[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                image_points.append((x, y))

                cv2.circle(
                    frame, (x, y), 3, (255, 0, 255), -1
                )

            image_points = np.array(
                image_points,
                dtype=np.float64
            )

            focal_length = width
            center = (width / 2, height / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            distortion = np.zeros((4, 1))

            success, rotation_vector, translation_vector = (
                cv2.solvePnP(
                    FACE_MODEL,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            )

            if success:

                rotation_matrix, _ = cv2.Rodrigues(
                    rotation_vector
                )

                angles = cv2.RQDecomp3x3(
                    rotation_matrix
                )[0]

                pitch = angles[0]
                yaw = angles[1]

                # Smooth head movements
                angle_history.append((pitch, yaw))

                smooth_pitch = np.mean([
                    angle[0] for angle in angle_history
                ])

                smooth_yaw = np.mean([
                    angle[1] for angle in angle_history
                ])

                # Calibrate initial neutral position
                if neutral_pitch is None:
                    neutral_pitch = smooth_pitch
                    neutral_yaw = smooth_yaw

                pitch_change = smooth_pitch - neutral_pitch
                yaw_change = smooth_yaw - neutral_yaw

                # Keep the tested left/right mapping
                if yaw_change < -10:
                    direction = "Turned Right"

                elif yaw_change > 10:
                    direction = "Turned Left"

                elif pitch_change < -15:
                    direction = "Looking Down"

                elif pitch_change > 15:
                    direction = "Looking Up"

                else:
                    direction = "Facing Forward"
                
                current_time = time.time()
                elapsed_time = current_time - last_frame_time
                last_frame_time = current_time

# Avoid counting unusually large time gaps
                elapsed_time = min(elapsed_time, 0.2)

                if direction == "Facing Forward":
                  facing_screen_time += elapsed_time

                elif direction in [
                  "Turned Left",
                   "Turned Right",
                   "Looking Up",
                   "Looking Down"
                  ]:
                  looking_away_time += elapsed_time


                total_tracked_time = facing_screen_time + looking_away_time
               
                if total_tracked_time > 0:
                    focus_score = (facing_screen_time / total_tracked_time) * 100
                else:
                  focus_score = 0     
                   # Send live monitoring data to Flask
                with data_lock:
                    monitor_data["focus_score"] = round(focus_score, 1)
                    monitor_data["blink_count"] = blink_count
                    monitor_data["eye_status"] = eye_status
                    monitor_data["head_direction"] = direction

                    monitor_data["focused_time"] = round(facing_screen_time, 1)
                    monitor_data["distracted_time"] = round(looking_away_time, 1)
                    monitor_data["drowsy_count"] = drowsy_count

                    focus_history.append({
                       "time": time.strftime("%H:%M:%S"),
                        "score": round(focus_score, 1)
                      })


        else:
            # Reset smoothing when the face disappears
            angle_history.clear()


        # ======================================
        # 8. ESTIMATE FOCUS STATUS
        # ======================================

        if direction == "Face not detected":
            focus_status = "Face Not Detected"

        elif direction == "Facing Forward":
            focus_status = "Facing Screen"

        else:
            focus_status = "Looking Away"

        # ======================================
        # 9. DISPLAY ATTENTIONAI DASHBOARD
     
        # ======================================

        cv2.putText(
            frame,
            "AttentionAI - Study Focus Monitor",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Eye Status: {eye_status}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if eye_status == "Eyes Open"
            else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blink Count: {blink_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Head Direction: {direction}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 0),
            2
        )

        cv2.putText(
            frame,
            f"Focus Status: {focus_status}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Screen-Facing Score appears ABOVE EAR
        cv2.putText(
            frame,
            f"Screen-Facing Score: {focus_score:.1f}%",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # EAR appears BELOW Screen-Facing Score
        cv2.putText(
            frame,
            f"EAR: {ear_value:.2f}",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        cv2.putText(
            frame,
            f"Yaw Change: {yaw_change:.1f}",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )
                # Save the annotated webcam frame for the Flask dashboard
        import camera_stream

        with camera_stream.frame_lock:
            camera_stream.latest_frame = frame.copy()

        cv2.imshow(
            "AttentionAI - Study Focus Monitor",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ==========================================
# 10. RELEASE RESOURCES
# ==========================================

cap.release()
face_mesh.close()
cv2.destroyAllWindows()