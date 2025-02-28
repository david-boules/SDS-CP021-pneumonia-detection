import os
import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.title("Medical X-Ray Imaging - Pneumonia Detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

model = load_model(MODEL_PATH)

image = st.file_uploader("Upload a Chest X-Ray Image", type=["png", "jpg", "jpeg"])

def image_preprocessing(image):
    new_image = Image.open(image)  # Convert UploadedFile to Image object
    new_image = new_image.convert("L")  # Convert to grayscale
    new_image = new_image.resize((128, 128))  # Resize to match model input
    
    new_image = np.array(new_image, dtype=np.float32) / 255.0  # Normalize

    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension
    new_image = np.expand_dims(new_image, axis=-1)  # Ensure shape is (1, 128, 128, 1)

    return new_image

if image is not None:
    st.subheader("Uploaded X-Ray Image:")
    st.image(image, caption="Chest X-Ray", width=500)

    if st.button("Predict"):
        preprocessed_image = image_preprocessing(image)
        if preprocessed_image is None:
            st.error("Error: Preprocessed image is None. Check preprocessing function.")
        else:
            prediction = model.predict(preprocessed_image)
            result = 'Pneumonia' if prediction[0][0] > 0.5 else "Normal"
            st.subheader("Prediction:")
            st.success(result)
            st.write(f"Prediction Score: {prediction[0][0]:,.2}")
            st.write(f"Confidence: {max(prediction[0][0], 1 - prediction[0][0]):,.2%}")

