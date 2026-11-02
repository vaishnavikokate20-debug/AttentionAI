
import cv2

# Access the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam connected successfully!")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the webcam frame.")
            break

        cv2.imshow("AttentionAI - Webcam Test", frame)

        # Press Q to close the webcam window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
