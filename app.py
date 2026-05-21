import streamlit as st
import numpy as np
from PIL import Image
from utils.face_detector import detect_faces
from utils.eye_detector import detect_eyes

st.title("Face Detection App")

st.write(
    "This app detects faces or eyes in uploaded images using OpenCV Haar Cascade classifiers."
)

detection_type = st.sidebar.selectbox(
    "Choose detection type",
    ["Face Detection", "Eye Detection"]
)

min_neighbors = st.sidebar.slider(
    "Detection strictness",
    min_value=3,
    max_value=15,
    value=10
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if detection_type == "Face Detection":
        result_img, count = detect_faces(img_array, min_neighbors)
    else:
        result_img, count = detect_eyes(img_array, min_neighbors)
    st.success(f"Detection completed. Found {count} object(s).")
    st.image(
        result_img,
        caption=f"Detected: {count}",
        use_container_width=True
    )
    