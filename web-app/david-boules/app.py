import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

st.title("Medical X-Ray Imaging - Pneumonia Detection")

model = load_model("model.keras")

image = st.file_uploader("Upload a Chest X-Ray Image", type=["png", "jpg", "jpeg"])

def image_preprocessing(image):
    new_image = Image.open(image) #image initially is of type 'UploadedFile' - this command turns it an 'Image' object
    new_image = new_image.resize((128,128))
    new_image = img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)
    new_image = new_image/255.0
    return new_image


if st.button("Predict"):
    preprocessed_image = image_preprocessing(image)
    if preprocessed_image is None:
        st.error("Error: Preprocessed image is None. Check preprocessing function.")
    else:
        prediction = model.predict(preprocessed_image)
        result = 'Pneumonia' if prediction >0.5 else "Normal"
        st.subheader("Prediction:")
        st.success(result)
        st.write(f"Prediction Score: {prediction[0][0]:,.2}")
        st.write(f"Confidence: {max(prediction[0][0], 1-prediction[0][0]):,.2%}")
        st.subheader("X-Ray Image:")
        st.image(image=preprocessed_image, caption=result, width=500)
