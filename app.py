import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load your trained YOLOv8 model (model file in same folder)
model = YOLO('best.pt')

st.title("Helmet Detection App 🪖")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read uploaded image with PIL
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(img_np)

    # Check detections count
    num_detections = results[0].boxes.xyxy.shape[0]

    if num_detections > 0:
        st.success(f"Helmet detected! 🪖 Total detections: {num_detections}")
    else:
        st.warning("No helmet detected.")

    # Get annotated image with detections drawn
    annotated_img = results[0].plot()

    # Convert annotated image back to PIL Image for display
    annotated_img = Image.fromarray(annotated_img)

    st.image(annotated_img, caption="Detection Results", use_column_width=True)
