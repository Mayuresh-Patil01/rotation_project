import streamlit as st
import requests

st.title("Face Rotation Classifier")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=files)
    if response.status_code == 200:
        st.write(f"Predicted Rotation: {response.json()['class']}")
    else:
        st.write("Error in prediction")