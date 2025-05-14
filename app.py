from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from training_testing import setup_model
from utils import mp_holistic, extract_keypoint_values, draw_styled_landmarks, mediapipe_detection, actions

app = Flask(__name__)

# Model ve değişkenleri global olarak tanımlayalım
model = setup_model()
model.load_weights('action.h5')
sequence = []
sentence = []
predictions = []
threshold = 0.5

def generate_frames():
    global sequence, sentence, predictions
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoint_values(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # --- BURADA HİÇBİR ŞEY YAZMA! ---
            # cv2.rectangle ve cv2.putText satırlarını kaldırdık

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_last_action')
def get_last_action():
    if sentence:
        return jsonify({'action': sentence[-1]})
    else:
        return jsonify({'action': 'Bekleniyor...'})

if __name__ == '__main__':
    app.run(debug=True) 
