import gradio as gr
import os 
import numpy as np 
import pandas as pd 

from PIL import Image
from tensorflow.keras.models import load_model
from huggingface_hub import snapshot_download, hf_hub_download

###### LOCAL SETTINGS #######
'''
def loadModel(): 
  model_path = os.path.join(os.path.dirname(__file__), 'models', 'trained_model.keras')   
  model = load_model(model_path)
  
  return model
'''

###### LOCAL HUGGINGFACE SETTINGS #######

# Replace with your model ID from HuggingFace Hub
MODEL_ID = 'ohinson1/cnn_pneumonia_detection'

def loadModel(): 
  # Load model from HuggingFace Hub with authentication
  try: 
    '''
    # Download/get the model from cache
    model_path = snapshot_download(
      repo_id=MODEL_ID,
      token=HF_TOKEN, 
      allow_patterns=["*"],  # Download all files
      local_files_only=False  # Set to True if you want to only use cache
    )
    print(f"Model downloaded to: {model_path}")
        
    # List the contents of the directory to see what files are available
    print("Available files:", os.listdir(model_path))
        
    # Create TFSMLayer with the full path
    model = TFSMLayer(
      model_path,
      call_endpoint='serving_default'
    )
    '''
    # Download/get the model from cache
    cache_dir = snapshot_download(
      repo_id=MODEL_ID,
      allow_patterns=["*"]
    )
    
    # Construct the full path to the .keras file
    model_path = os.path.join(cache_dir, 'trained_model.keras')
    print(f"Loading model from: {model_path}")
    
    # Load the .keras file directly
    model = load_model(model_path)
    return model
  except Exception as e: 
    print(f'Error loading model: {str(e)}')
    raise

###### HUGGINGFACE SETTINGS #######
'''
# Replace with your model ID from HuggingFace Hub
MODEL_ID = 'ohinson1/cnn_pneumonia_detection'
# HuggingFace token - preferably loaded from environment variable
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')  # Set this in your environment variables

# Load model globally
if not HF_TOKEN:
    print("Warning: HUGGINGFACE_TOKEN environment variable not set")
    print("If this is a private repository, you need to set your token.")
    print("You can either:")
    print("1. Set the HUGGINGFACE_TOKEN environment variable")
    print("2. Use huggingface-cli login")
    print("3. Modify the code to directly pass your token (not recommended for production)")

def loadModel(): 
  # Load model from HuggingFace Hub with authentication
  try: 
    # Download the model file from Hugging Face Hub
    # Use this since CNN not a recognized model type in HuggingFace's 
    # transformer library 
    model_path = hf_hub_download(
      repo_id=MODEL_ID, 
      token=HF_TOKEN, 
      filename='trained_model.keras'
    )
    # Load the model
    model = load_model(model_path)
    return model
  except Exception as e: 
    print(f'Error loading model: {str(e)}')
    print('Please ensure you have provided a valid HuggingFace token for private repository access')
    raise
'''

model = loadModel()

def classify_images(img): 
  if img is None: 
    raise ValueError("No image was uploaded")
    
  # Convert to PIL Image if receiving numpy array
  if isinstance(img, np.ndarray):
    img = Image.fromarray(img)
  
  class_names = {
      0: 'NORMAL', 
      1: 'PNEUMONIA'
    }
    
  # Resize using PIL
  img = img.resize((180, 180))
  # Convert to numpy and reshape
  img_array = np.array(img)
  img_array = img_array.reshape((-1, 180, 180, 3))
  
  prediction = model.predict(img_array).flatten()
  # Convert prediction to class labels
  label = int(prediction[0] >= 0.5)  # Assuming a binary classifier (0: NORMAL, 1: PNEUMONIA)
    
  # Format output for Gradio
  return {class_names[label]: float(prediction[0])}

markdown = '''
    # Detecting Pneumonia From Chest X-Rays

'''

with gr.Blocks(theme=gr.themes.Citrus(), fill_height=True) as app: 
  gr.Markdown(markdown)
  with gr.Row(): 
    with gr.Column(scale=1): 
      upload_image = gr.Image(type='pil', width=224, height=224)
      predict_btn = gr.Button('Detect')
    with gr.Column(scale=2):
      output_prediction = gr.Label(label='Prediction')
  predict_btn.click(fn=classify_images, inputs=upload_image, outputs=output_prediction, api_name='prediction')

if __name__ == "__main__":  
  app.launch()