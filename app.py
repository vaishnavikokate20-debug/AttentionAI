
from flask import Flask, render_template, jsonify, Response
from threading import Lock
from collections import deque
import cv2
import time

app = Flask(__name__)

# Shared monitoring data
monitor_data = {
    "focus_score": 0.0,
    "blink_count": 0,
    "eye_status": "Waiting",
    "head_direction": "Waiting",
    "focused_time": 0.0,
    "distracted_time": 0.0,
    "drowsy_count": 0
}

# Protect shared data when Flask and the webcam update it
data_lock = Lock()

# Store the most recent 60 focus-score readings
focus_history = deque(maxlen=60)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate_frames():
        import camera_stream

        while True:
            with camera_stream.frame_lock:
                if camera_stream.latest_frame is None:
                    frame = None
                else:
                    frame = camera_stream.latest_frame.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            success, buffer = cv2.imencode(".jpg", frame)

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            time.sleep(0.03)

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def get_data():
    with data_lock:
        return jsonify(monitor_data)


@app.route("/focus_history")
def get_focus_history():
    with data_lock:
        return jsonify(list(focus_history))


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)