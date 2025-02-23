import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load trained model
class CNNClassifierGrayscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Load the trained model
model = CNNClassifierGrayscale().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Function to predict image
def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence_score = torch.max(probabilities)
    predicted_label = 'PNEUMONIA' if predicted_class.item() == 1 else 'NORMAL'
    return predicted_label, confidence_score.item()

# Streamlit UI
st.set_page_config(layout="centered")
st.title("⚕️ Pneumonia Detection Using CNN")

# Sidebar Information
st.sidebar.title("📌 About this Project")
st.sidebar.write(
    "### 🌍 A Collaborative Effort"
    "\nThis project is a collaborative initiative brought to you by **SuperDataScience**, a thriving community dedicated "
    "to advancing the fields of data science, machine learning, and AI."
    "\n\n### 🔬 Project Overview"
    "\nWe have built a **Convolutional Neural Network (CNN)** model to classify medical X-ray images and detect pneumonia."
)

# File uploader
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption=" Uploaded Image", width=300)
    
    # Save uploaded file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict
    predicted_class, confidence = predict_image("temp.jpg", model, device)
    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f" Confidence Score: **{confidence:.4f}**")
