# app.py — Ride Demand Forecasting App
# ------------------------------------

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ------------------------------------------------
# Load trained model from gold layer
# ------------------------------------------------
MODEL_PATH = "data/gold/ride_demand_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully from Gold Layer!")
else:
    st.error("❌ Model not found! Please train and save it first.")
    st.stop()

# ------------------------------------------------
# App title and description
# ------------------------------------------------
st.title("🚖 Ride Demand Forecasting System")
st.markdown("""
### Predict Ride Demand based on Time & Day  
This app forecasts the number of ride requests (cab/auto) for a given time period.
""")

# ------------------------------------------------
# Input section
# ------------------------------------------------
st.subheader("Enter Time Details:")

col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Select Hour of the Day (0–23):", 0, 23, 8)
with col2:
    day = st.selectbox(
        "Select Day of Week:",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

# Create input DataFrame
input_data = pd.DataFrame({
    'hour': [hour],
    'day': [day]
})

# Convert day to numerical encoding (if needed)
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
input_data['day'] = input_data['day'].map(day_map)

# ------------------------------------------------
# Predict
# ------------------------------------------------
if st.button("🔮 Predict Ride Demand"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🚗 Estimated Ride Demand: **{int(prediction)} rides/hour**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.caption("Developed by SARAVANA A, UDHAYANIDHI D, SHYAM PRASAD S — Department of AIDS")
