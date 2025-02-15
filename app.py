import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


# Define custom functions
def is_north(X):
    north_cities = ['Delhi', 'New Delhi', 'Kolkata']
    return (
        X.assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in X.columns
        }).drop(columns=X.columns)
    )


def part_of_day(X, morning=4, noon=12, evening=16, night=20):
    columns = X.columns.to_list()
    X_Temp = X.assign(**{
        col: pd.to_datetime(X.loc[:, col]).dt.hour
        for col in columns
    })
    return (
        X_Temp.assign(**{
            f"{col}_part_of_day": np.select(
                [X_Temp.loc[:, col].between(morning, noon, inclusive='left'),
                 X_Temp.loc[:, col].between(noon, evening, inclusive='left'),
                 X_Temp.loc[:, col].between(evening, night, inclusive='left')],
                ['morning', 'afternoon', 'evening'],
                default='night'
            )
            for col in columns
        }).drop(columns=X.columns)
    )


def is_direct(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=['total_stops'])
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))


# Load model
model = joblib.load("saved_models/model1.joblib")

train = pd.read_csv("Data/Train.csv")
X_Train = train.drop(columns='price')
y_Train = train.price.copy()

# Streamlit UI
st.title("✈️ Flight Price Prediction App")
st.markdown("### Enter flight details to predict the price")
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f4f4f4; /* Light blue */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
with st.sidebar:
    st.header("🔹 Flight Details")
    airline = st.selectbox("✈️ Airline", options=X_Train.airline.unique())
    date_of_journey = st.date_input("📅 Date of Journey")
    source = st.selectbox("📍 Source", options=X_Train.source.unique())
    filtered_destinations = X_Train.destination[
        (X_Train.destination != source) & (X_Train.destination != "New Delhi")
        ].unique()
    destination = st.selectbox("📍 Destination", options=filtered_destinations)
    dep_time = st.time_input("🕒 Departure Time")
    departure_datetime = datetime.combine(date_of_journey, dep_time)
    arrival_time = st.time_input("🕒 Arrival Time")
    arrival_datetime = datetime.combine(date_of_journey, arrival_time)
    duration = st.number_input("⏳ Duration (hours)")
    total_stops = st.number_input("🔀 Total Stops", min_value=0, max_value=5, step=1)
    additional_info = st.selectbox("ℹ️ Additional Info", options=X_Train.additional_info.unique())
    flight_class = st.selectbox("🎟️ Class", options=X_Train.class_.unique())

if st.button("🔮 Predict Price"):
    # Convert inputs into a DataFrame
    input_data = pd.DataFrame({
        'airline': [airline],
        'date_of_journey': [date_of_journey],
        'source': [source],
        'destination': [destination],
        'dep_time': [departure_datetime],
        'arrival_time': [arrival_datetime],
        'duration': [duration],
        'total_stops': [total_stops],
        'additional_info': [additional_info],
        'class': [flight_class]
    })

    # Apply preprocessing
    transformed_data = model.named_steps['pre'].transform(input_data)

    # Make prediction
    predicted_price = model.named_steps['XG Boost'].predict(transformed_data)[0]

    st.markdown(
        f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: #d0f0c0;">
            <h2 style="color: blue;">Predicted Flight Price</h2>
            <h1 style="color: green;">₹{predicted_price:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )