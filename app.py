import streamlit as st
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion detection model
emotion_model = load_model('/content/emotion_model.hdf5')

# Emotion labels based on the model's training data
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        return roi_gray, (x, y, w, h)
    return None, None

# Streamlit app layout
st.title("Real-time Emotion Detection")
run = st.checkbox('Run')
frame_window = st.image([])

camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    face_img, bbox = preprocess_frame(frame)
    if face_img is not None:
        prediction = emotion_model.predict(face_img)
        emotion_label = emotion_labels[np.argmax(prediction)]

        # Draw the bounding box and label
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    frame_window.image(frame, channels='BGR')
else:
    st.write('Stopped')
    camera.release()
