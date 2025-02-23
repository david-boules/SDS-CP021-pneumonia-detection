import os
import torch
import streamlit as st

st.write("Current working directory:", os.getcwd())

MODEL_PATH = "./model.pth"
if os.path.exists(MODEL_PATH):
    st.success("Model file found!")
else:
    st.error("Model file NOT found! Please check file path.")
