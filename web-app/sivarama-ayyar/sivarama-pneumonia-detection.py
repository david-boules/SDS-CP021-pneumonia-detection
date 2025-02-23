import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
# Page configuration
st.set_page_config(layout="wide")


@st.cache_resource
def load_model():
    """
    Loads the pre-trained model

    Returns:
    model: The pre-trained model.
    """
    model = tf.keras.models.load_model('/Users/sivaram/Developer/SDS/SuperDataScience-Community-Projects/SDS-CP021-pneumonia-detection/web-app/sivarama-ayyar/sivarama-pneumonia_detection-CNN.keras')
    #model = joblib.load("/Users/sivaram/Developer/SDS/SuperDataScience-Community-Projects/SDS-CP021-pneumonia-detection/web-app/sivarama-ayyar/sivarama-pneumonia_detection-CNN.keras")
    return model


with st.spinner('Model is being loaded..'):
    model=load_model()


def import_and_predict(image, model):
    image_data =  load_image(image)
    prediction = model.predict(image_data)
    pred_prob = prediction[0][0]
    pred_class = 1 if pred_prob > 0.5 else 0
    pred_label = "Pneumonia" if pred_class == 1 else "Normal"
    confidence = pred_prob if pred_class == 1 else 1 - pred_prob
    return pred_label, confidence

def load_image(image):
    if image.mode != "L":
        image = image.convert("L")  # Convert to grayscale
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.repeat(image_array, 3, axis=-1)  # Convert grayscale to 3 channels
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Expand batch dimension
    return image_array

# Main app function
def main():
    st.title("Pneumonia Detection using CNN pretrained model")
    upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
    if upload is None:
        st.text("Please upload an image file")
    else:
        image= Image.open(upload)
        st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
        pred_label, confidence = import_and_predict(image, model)
        print(pred_label, confidence)
        string = "Detected Disease : " + pred_label
        if pred_label == 'Normal':
            st.balloons()
            st.sidebar.success('NORMAL')
        elif pred_label == 'Pneumonia':
            st.sidebar.warning(string)
        st.write(f"(Confidence: {confidence:.2%})")


# Run the app
if __name__ == "__main__":
      main()
