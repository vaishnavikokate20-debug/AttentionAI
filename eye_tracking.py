import cv2
import mediapipe as mp
import math
import time

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20
MIN_BLINK_DURATION = 0.05
MAX_BLINK_DURATION = 0.50

blink_count = 0
eyes_closed = False
closed_start_time = None


def calculate_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def calculate_ear(eye_points):
    vertical1 = calculate_distance(eye_points[1], eye_points[5])
    vertical2 = calculate_distance(eye_points[2], eye_points[4])
    horizontal = calculate_distance(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2 * horizontal)


if not cap.isOpened():
    print("Error: Could not access webcam.")

else:
    print("Blink detection started! Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        eye_status = "Face not detected"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_points = []
                right_points = []

                for index in LEFT_EYE:
                    landmark = face_landmarks.landmark[index]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    left_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                for index in RIGHT_EYE:
                    landmark = face_landmarks.landmark[index]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    right_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                left_ear = calculate_ear(left_points)
                right_ear = calculate_ear(right_points)
                average_ear = (left_ear + right_ear) / 2

                current_time = time.time()

                if average_ear < EAR_THRESHOLD:
                    eye_status = "Eyes Closed"

                    if not eyes_closed:
                        eyes_closed = True
                        closed_start_time = current_time

                else:
                    eye_status = "Eyes Open"

                    if eyes_closed:
                        duration = current_time - closed_start_time

                        if MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                            blink_count += 1

                        eyes_closed = False
                        closed_start_time = None

                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    eye_status,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if eye_status == "Eyes Open"
                    else (0, 0, 255),
                    2
                )

        cv2.putText(
            frame,
            f"Blink Count: {blink_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("AttentionAI - Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()