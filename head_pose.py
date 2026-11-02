import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Open webcam
cap = cv2.VideoCapture(0)

# Facial landmark indexes
LANDMARKS = [1, 152, 33, 263, 61, 291]

# Approximate 3D face model
FACE_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

# Store recent angles to smooth sudden changes
angle_history = deque(maxlen=10)

neutral_pitch = None
neutral_yaw = None

if not cap.isOpened():
    print("Error: Could not access webcam.")

else:
    print("Head pose estimation started. Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read webcam frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        direction = "Face not detected"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            image_points = []

            # Extract selected facial landmarks
            for index in LANDMARKS:
                landmark = face_landmarks.landmark[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                image_points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            image_points = np.array(
                image_points,
                dtype=np.float64
            )

            # Camera parameters
            focal_length = width
            center = (width / 2, height / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            distortion = np.zeros((4, 1))

            # Estimate face rotation
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
                print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")
                


                # Smooth the angle values
                angle_history.append((pitch, yaw))

                smooth_pitch = np.mean([
                    angle[0] for angle in angle_history
                ])

                smooth_yaw = np.mean([
                    angle[1] for angle in angle_history
                ])

                # Calibrate using the initial forward-facing position
                if neutral_pitch is None:
                    neutral_pitch = smooth_pitch
                    neutral_yaw = smooth_yaw

                # Measure changes relative to the initial position
                pitch_change = smooth_pitch - neutral_pitch
                yaw_change = smooth_yaw - neutral_yaw
                print(f"Yaw Change: {yaw_change:.2f}")

                # Classify direction using angle thresholds
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

                # Display direction and angles ONCE
                cv2.putText(
                    frame,
                    direction,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Yaw Change: {yaw_change:.1f} deg",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Pitch Change: {pitch_change:.1f} deg",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        else:
            angle_history.clear()

        # Show webcam window
        cv2.imshow(
            "AttentionAI - Head Pose Estimation",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()