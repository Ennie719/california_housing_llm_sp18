
from fastapi import FastAPI
from preprocess import load_config, load_data, load_data, check_missing_values, check_duplicates, splitting_data, scale_features, feature_target_corr
import joblib
import sys
import os
# make sure the repo root is importable when Streamlit runs this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
# from api.schemas import PredictionResponse
from api.api import make_prediction
st.set_page_config(page_title="California Housing Predictor", page_icon="🏠")
st.title("California Housing Predictor")
st.caption("Describe your housing details in plain English. This is necessary for accurate predictions.")



if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Tell me about your housing details, for example: "
                "For example: "
                "\"Predict the price of the house in california of a 3-bedroom, build year 2020."
             "\""
             )
        }
    ]
for msg in st.session_state.messages:
    
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
user_text = st.chat_input("Describe your housing details here...")
if user_text:
    st.session_state.messages.append(
                            {"role": "user",
                                      "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner(" Analyzing ..."):
            # data = load_data()
            result = make_prediction()
            reply = f"The predicted price of the house is ${result}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    # For example:
    # prediction = predict(user_text)
    # st.chat_message("assistant").markdown(f"Prediction: {prediction['prediction']}")     