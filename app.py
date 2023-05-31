import numpy as np
import cv2
from flask import Flask, render_template, Response, request

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')

selected_glasses = "Glasses3.1.png"  # Default glasses image

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        text = "To try on different glasses, please navigate to the previous page using the arrow at the top left"
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_font_scale = 0.4
        text_thickness = 1
        text_color = (0, 0, 0)  # Red color
        text_size, _ = cv2.getTextSize(text, text_font, text_font_scale, text_thickness)
        rect_x = 10
        rect_y = frame.shape[0] - text_size[1] - 20
        rect_width = frame.shape[1] - 20
        rect_height = text_size[1] + 10
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)
        cv2.putText(frame, text, (rect_x, rect_y + text_size[1] + 5), text_font, text_font_scale, text_color, text_thickness, cv2.LINE_AA)




        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
                resized_img = cv2.resize(img_to_place, (ew, eh))
                mask = resized_img[:, :, 3] / 255.0
                mask = cv2.merge((mask, mask, mask))
                resized_img = resized_img[:, :, 0:3]  # Remove the alpha channel
                background_roi = roi_color[ey:ey+eh, ex:ex+ew, 0:3]  # Remove the alpha channel
                blended_img = cv2.multiply(resized_img.astype(float), mask) + cv2.multiply(background_roi.astype(float), 1.0 - mask)
                blended_img = blended_img.astype(np.uint8)
                roi_color[ey:ey+eh, ex:ex+ew, 0:3] = blended_img

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        scale_factor = 1.8
        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__ == '__main__':
    app.run(debug=True)
