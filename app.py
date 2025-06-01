import streamlit as st
from PIL import Image
import torch

# ========================
# Set Page Config
# ========================import streamlit as st
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

st.set_page_config(
    page_title="Helmet Detection App",
    layout="centered",
    initial_sidebar_state="auto"
)

# ========================
# Background Image Styling
# ========================
background_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1610891903274-6b39d2091d16?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.4);
}
</style>
"""

st.markdown(background_img, unsafe_allow_html=True)

# ========================
# Title & Subtitle
# ========================
st.markdown("<h1 style='text-align: center; color: white;'>🚧 Helmet Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Upload an image and detect if a helmet is present using YOLOv5</p>", unsafe_allow_html=True)

# ========================
# Sidebar Info
# ========================
st.sidebar.title("📌 About")
st.sidebar.markdown(
    """
This app uses a YOLOv5 model to detect helmets in uploaded images.

- Built with Streamlit
- Powered by PyTorch
"""
)

# ========================
# Image Upload
# ========================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 Uploaded Image', use_column_width=True)

    st.markdown("<h4 style='color:white;'>🔍 Running detection...</h4>", unsafe_allow_html=True)

    # ========================
    # Load your YOLO model
    # ========================
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    results = model(image)

    # ========================
    # Show results
    # ========================
    st.image(results.render()[0], caption="🧠 Detection Result", use_column_width=True)

    # Optional: Text Output
    labels = results.pandas().xyxy[0]['name'].value_counts().to_dict()
    st.markdown(f"<h5 style='color:white;'>📝 Detection Summary</h5>", unsafe_allow_html=True)
    if labels:
        for label, count in labels.items():
            st.write(f"- **{label}**: {count}")
    else:
        st.write("❌ No objects detected.")

