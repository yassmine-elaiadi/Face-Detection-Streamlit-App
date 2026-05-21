import streamlit as st

st.title("About This Project")

st.write("""
This project is a computer vision web app built with Streamlit and OpenCV.

It can:
- detect faces in uploaded images
- detect eyes in uploaded images
- use webcam image capture
- adjust detection strictness

The app uses Haar Cascade classifiers, which are pre-trained OpenCV models.
""")