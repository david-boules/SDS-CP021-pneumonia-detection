import gradio as gr
import os 
import numpy as np 
import pandas as pd 

from PIL import Image
from tensorflow.keras.models import load_model

def loadModel(): 
  model_path = os.path.join(os.path.dirname(__file__), 'models', 'trained_model.keras')   
  model = load_model(model_path)
  
  return model 

model = loadModel()

def classify_images(img): 
  if img is None: 
    raise ValueError("No image was uploaded")
    
  # Convert to PIL Image if receiving numpy array
  if isinstance(img, np.ndarray):
    img = Image.fromarray(img)
  
  class_names = {
      '0': 'NORMAL', 
      '1': 'PNEUMONIA'
    }
    
  # Resize using PIL
  img = img.resize((180, 180))
  # Convert to numpy and reshape
  img_array = np.array(img)
  img_array = img_array.reshape((-1, 180, 180, 3))
  
  prediction = model.predict(img_array).flatten()
  labels = (prediction < 0.5).astype(int)
  confidences = {labels[i]: float(prediction[i]) for i in range(1)}
  return confidences 

with gr.Blocks() as app: 
  upload_image = gr.Image(type='pil', width=224, height=224)
  output_prediction = gr.Label()
  predict_btn = gr.Button('Detect')
  predict_btn.click(fn=classify_images, inputs=upload_image, outputs=output_prediction, api_name='prediction')
  
if __name__ == "__main__":  
  app.launch()