import streamlit as st
import requests

st.title("Spam Message Classifier")

message = st.text_area("Enter the message:")

if st.button("Predict"):
    if message:
        response = requests.post("http://api:80/predict", json={"text": message})
        result = response.json()
        st.write(f"Prediction: {result['prediction']}")
    else:
        st.write("Please enter a message.")
