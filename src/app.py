 
import json
import os
import sys
import openai
import streamlit as st

# Ensure the repository root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your prediction function from the API layer
from api.api import make_prediction

st.set_page_config(page_title="California Housing Predictor", page_icon="🏠")
st.title("🏠 California Housing Predictor")
st.caption("Describe your housing details in plain English. The AI will extract the parameters and run the prediction.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Tell me about your housing details, for example:\n"
                "\"Predict the price of a 3-bedroom California house built in 2020 with a median income of $60,000.\""
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def parse_user_text_with_llm(text: str) -> dict:
    """Uses an LLM to convert natural language text into structured California Housing features."""
    system_prompt = """
    Extract California Housing dataset features from the user prompt into a valid JSON object.
    Expected schema keys:
    - "MedInc": float (median income in tens of thousands, e.g. $50k = 5.0)
    - "HouseAge": float
    - "AveRooms": float
    - "AveBedrms": float
    - "Population": float
    - "AveOccup": float
    - "Latitude": float
    - "Longitude": float
    
    Use sensible defaults if fields are missing: 
    {"MedInc": 3.5, "HouseAge": 20, "AveRooms": 5, "AveBedrms": 1, "Population": 1000, "AveOccup": 3, "Latitude": 34.0, "Longitude": -118.0}
    Return ONLY valid JSON.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        # Fallback default dictionary if LLM call fails
        return {
            "MedInc": 3.5, "HouseAge": 20, "AveRooms": 5, 
            "AveBedrms": 1, "Population": 1000, "AveOccup": 3, 
            "Latitude": 34.0, "Longitude": -118.0
        }

user_text = st.chat_input("Describe your housing details here...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing text with LLM and running prediction..."):
            # 1. Parse natural language into features dict using LLM
            extracted_features = parse_user_text_with_llm(user_text)
            
            # 2. Pass structured features into your prediction function
            result = make_prediction(extracted_features)
            
            # Format output (assuming result returns USD or scaled value)
            reply = f"The estimated price of the house is **${result:,.2f}**."
            
        st.markdown(reply)
        
    st.session_state.messages.append({"role": "assistant", "content": reply})