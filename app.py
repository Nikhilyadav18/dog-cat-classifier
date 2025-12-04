import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

MODEL_PATH = "models/mobilenetv2_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)

st.title("🐶🐱 Dog vs Cat Classifier (MobileNetV2)")
st.write("Upload an image and the model will classify it as Dog or Cat.")

def predict_from_array(img_array):
    img = cv2.resize(img_array, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "Dog 🐶" if pred > 0.5 else "Cat 🐱"

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Uploaded Image", width=300)

    img_array = np.array(image)

    st.write("### Predicting...")
    label = predict_from_array(img_array)
    st.success(f"Prediction: {label}")
