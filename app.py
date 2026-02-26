import streamlit as st
import pickle
import pandas as pd
from src.feature_engineering import add_features

st.title("🍽 Kitchen Prep Time Prediction System")

# Load model
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# User Inputs
num_items = st.number_input("Number of Items", 1, 20)
total_quantity = st.number_input("Total Quantity", 1, 50)
order_value = st.number_input("Order Value", 100, 3000)
restaurant_load = st.number_input("Restaurant Load", 1, 100)
avg_prep_time = st.number_input("Average Prep Time", 5, 60)
staff_available = st.number_input("Staff Available", 1, 30)
is_weekend = st.selectbox("Weekend?", [0, 1])
is_rain = st.selectbox("Rain?", [0, 1])
is_peak_hour = st.selectbox("Peak Hour?", [0, 1])

if st.button("Predict Kitchen Prep Time"):

    input_data = {
        "num_items": num_items,
        "total_quantity": total_quantity,
        "order_value": order_value,
        "restaurant_load": restaurant_load,
        "avg_prep_time": avg_prep_time,
        "staff_available": staff_available,
        "is_weekend": is_weekend,
        "is_rain": is_rain,
        "is_peak_hour": is_peak_hour
    }

    df = pd.DataFrame([input_data])
    df = add_features(df)

    prediction = model.predict(df)[0]

    st.success(f"Predicted Kitchen Prep Time: {round(prediction, 2)} minutes")

    # Delay Logic
    if prediction > avg_prep_time + 5:
        st.error("⚠ High Delay Risk!")
    else:
        st.info("✅ Normal Prep Time")
