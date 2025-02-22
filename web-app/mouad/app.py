import os
import streamlit as st
import tensorflow as tf
import boto3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image as PILImage
import cv2
import numpy as np
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore

# AWS S3 details
BUCKET_NAME = "pneumonia-models"  # Replace with your S3 bucket name
MODEL_PATH = "model.h5"
LOCAL_MODEL_PATH = "model.h5"

# Download model from S3
def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.text("Downloading model from S3... Please wait.")
        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, MODEL_PATH, LOCAL_MODEL_PATH)
        st.text("Download complete!")

# Load Keras model
@st.cache_resource
def load_keras_model():
    download_model()
    return load_model(LOCAL_MODEL_PATH)

# Preprocess the image to match training input
def preprocess_image(image):
    if image.mode != "L":
        image = image.convert("L")  # Convert to grayscale
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.repeat(image_array, 3, axis=-1)  # Convert grayscale to 3 channels
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Expand batch dimension
    return image_array

# Get model prediction
def get_prediction(model, image_array):
    preds = model.predict(image_array)
    pred_prob = preds[0][0]
    pred_class = 1 if pred_prob > 0.5 else 0
    pred_label = "Pneumonia" if pred_class == 1 else "Normal"
    confidence = pred_prob if pred_class == 1 else 1 - pred_prob
    return pred_label, confidence

# Grad-CAM visualization
def generate_gradcam(model, image_array, pred_class):
    try:
        score = BinaryScore(pred_class == 1)
        model_modifier = ReplaceToLinear()
        gradcam = Gradcam(model, model_modifier=model_modifier, clone=False)
        cam = gradcam(score, image_array, penultimate_layer=-1)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam[0]), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = 0.5 * image_array[0] + 0.5 * heatmap / 255.0
        return overlay
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None

# Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-Rays")

# Upload File Section
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = PILImage.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)

    # Load the model and get prediction
    model = load_keras_model()
    image_array = preprocess_image(image)
    pred_label, confidence = get_prediction(model, image_array)

    # Display prediction result
    st.write(f"### Prediction: **{pred_label}** (Confidence: {confidence:.2%})")

    # Grad-CAM visualization
    if st.button("Show Grad-CAM Visualization"):
        gradcam_overlay = generate_gradcam(model, image_array, 1 if pred_label == "Pneumonia" else 0)
        if gradcam_overlay is not None:
            st.image(gradcam_overlay, caption="Grad-CAM Heatmap", use_column_width=True)
