import streamlit as st
import numpy as np
from PIL import Image
from utils.face_detector import detect_faces

st.title("Webcam Face Detection")

min_neighbors = st.sidebar.slider(
    "Detection strictness",
    min_value=3,
    max_value=15,
    value=10
)

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image)
    img_array = np.array(image)

    result_img, face_count = detect_faces(img_array, min_neighbors)

    st.image(
        result_img,
        caption=f"Detected Faces: {face_count}",
        use_container_width=True
    )