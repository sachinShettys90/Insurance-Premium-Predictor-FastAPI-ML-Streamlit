"""
Streamlit frontend for the Insurance Premium Prediction API.

Initial setup
1. Make sure your FastAPI server is running first:
    uvicorn app:app --reload

    (it should be live at http://127.0.0.1:8000)

2. Run this app:
    streamlit run frontend.py
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

TIER_1_CITIES = ["Mumbai", "Delhi", "Bangalore",
                 "Chennai", "Kolkata", "Hyderabad", "Pune"]
TIER_2_CITIES = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri",
]
ALL_CITIES = sorted(set(TIER_1_CITIES + TIER_2_CITIES))

OCCUPATIONS = [
    "retired", "unemployed", "business_owner", "government_job",
    "private_job", "freelancer", "student",
]

st.set_page_config(page_title="Insurance Premium Predictor")
st.title("Insurance Premium Category Predictor")
st.caption(
    "Fill in the details below to predict the premium category (Low / Medium / High).")


st.markdown("Enter your details below:")

# Input fields
with st.form("prediction_form"):
    st.markdown("Enter your details below:")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=119, value=30)
    weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
    height = st.number_input(
        "Height (m)", min_value=0.5, max_value=2.5, value=1.7)
    income_lpa = st.number_input(
        "Annual Income (LPA)", min_value=0.1, value=10.0)
    smoker = st.selectbox("Are you a smoker?", options=[True, False])
    city = st.selectbox("City", options=ALL_CITIES,
                        index=ALL_CITIES.index("Mumbai"))
    occupation = st.selectbox("Occupation", options=OCCUPATIONS)

    submitted = st.form_submit_button("Predict Premium Category")


if submitted:
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(
                f"Predicted Insurance Premium Category: **{result['prediction_category']}**")
        else:
            st.error(f"API ERROR : {response.status_code}-{response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            "could not connect to the FAST API server.Make sure its running on port 8000")
