import numpy as np
import cv2
from flask import Flask, render_template, Response, request

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')

selected_glasses = "Glasses1.1.png"  # Default glasses image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html', glasses=selected_glasses)

@app.route('/select_glasses', methods=['POST'])
def select_glasses():
    global selected_glasses
    selected_glasses = request.form['glasses']
    return '', 200

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img_to_place = cv2.imread('static/images/' + selected_glasses, cv2.IMREAD_UNCHANGED)
        # Rest of your code ...
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_to_place = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_h, img_w = gray.shape
        img_to_place_h, img_to_place_w, = gray_to_place.shape

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
                resized_img = cv2.resize(img_to_place, (ew, eh), interpolation=cv2.INTER_AREA)
                resized_img_h, resized_img_w, _ = resized_img.shape

                roi_color[ey:ey+resized_img_h, ex:ex+resized_img_w, :] = resized_img
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
