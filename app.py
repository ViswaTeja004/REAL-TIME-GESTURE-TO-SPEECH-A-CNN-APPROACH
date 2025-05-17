import streamlit as st
import pyttsx3
import numpy as np
import cv2
import mediapipe as mp
from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file 
from src.config import SEQ_LEN, THRESH_HOLD

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

s2p_map = {k.lower(): v for k, v in load_json_file("src/sign_to_prediction_index_map.json").items()}
p2s_map = {v: k for k, v in load_json_file("src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

models_path = ['./models/islr-fp16-192-8-seed_all42-foldall-last.h5']
models = [get_model() for _ in models_path]

for model, path in zip(models, models_path):
    model.load_weights(path)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def real_time_asl():
    res = []
    tflite_keras_model = TFLiteModel(islr_models=models)
    sequence_data = []
    cap = cv2.VideoCapture(0)
    
    # Updated CSS for styling (Box for Detected Signs)
    st.markdown("""
        <style>
            .title {
                text-align: left;
                color: red;
                font-size: 40px;
                font-weight: bold;
                position: relative;
                left: 0;
            }
            .detected-signs-box {
                border: 3px solid black; /* Box Border */
                padding: 15px;
                margin-top: 20px;
                font-size: 30px;
                font-weight: bold;
                background-color: #f9f9f9;
                border-radius: 10px;
                width: 50%;
            }
        </style>
        <div class="title">Hand gesture to text-speech recognition</div>
    """, unsafe_allow_html=True)

    placeholder = st.empty()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break
            
            image, results = mediapipe_detection(frame, holistic)
            draw(image, results)
            
            try:
                landmarks = extract_coordinates(results)
            except:
                landmarks = np.zeros((468 + 21 + 33 + 21, 3))
            
            sequence_data.append(landmarks)
            sign = ""
            
            if len(sequence_data) % SEQ_LEN == 0:
                prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]
                if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                    sign = np.argmax(prediction.numpy(), axis=-1)
                sequence_data = []
            
            if sign != "" and decoder(sign) not in res:
                detected_sign = decoder(sign)
                res.insert(0, detected_sign)
                speak(detected_sign)
            
            # Display detected signs inside a box
            placeholder.markdown(f"""
                <div class='detected-signs-box'>
                    Detected Signs:<br>{'<br>'.join(res)}
                </div>
            """, unsafe_allow_html=True)
            
            cv2.imshow('Webcam Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_asl()
