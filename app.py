import streamlit as st
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import tempfile
import os
import requests
import base64
from io import BytesIO
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="Helmet Detection App", page_icon="⛑️", layout="centered")

# Optional: Set helmet-related background image
def set_background():
    image_url = "https://images.unsplash.com/photo-1603521341605-d985d20e19af?auto=format&fit=crop&w=1950&q=80"
    response = requests.get(image_url)
    encoded = base64.b64encode(response.content).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Load your YOLOv8 model
model = YOLO("besttt.pt")

# App title
st.markdown("<h1 style='color:white; text-align:center;'>Helmet Detection</h1>", unsafe_allow_html=True)
st.write("Upload an image to detect helmets:")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Run YOLO model
    results = model(temp_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes
    image = Image.open(temp_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw boxes (no labels)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

    # Show image
    st.image(image, caption="Bounding boxes only", use_column_width=True)

    # Cleanup
    os.remove(temp_path)
