import streamlit as st
import pickle
import numpy as np
import datetime
from datetime import datetime
import pandas as pd
import math
from math import log
from explore_page import calculate_dewpoint

import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

with open ('knn_model.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    columns_knn = data['columns']

def predict_energy_from_input():
    """
    Streamlit function that:
      - Asks the user for date, time, temperature (°C) and humidity (%)
      - Calculates dew point using calculate_dewpoint()
      - Creates a DataFrame with a timestamp index
      - Generates dummy variables for the hour and day of week
      - Concatenates these with temperature, humidity, and dew point features
      - Uses the provided model to predict the energy load
    """
    st.title("Energy Load Prediction")

    # Get date and time inputs
    user_date = st.date_input("Select Date")
    user_time = st.time_input("Select Time")
    
    # Combine date and time into one timestamp
    timestamp = pd.to_datetime(f"{user_date} {user_time}")
    
    # Get temperature and humidity inputs
    temp = st.number_input("Temperature (°C)", value=20.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    
    # Calculate dew point
    dewpoint = calculate_dewpoint(temp, humidity)
    st.write(f"Calculated Dew Point: {dewpoint:.2f} °C")
    
    # Create a DataFrame with the continuous features and timestamp as index
    data = {'TemperatureC': [temp],
            'Humidity': [humidity],
            'Dew PointC': [dewpoint]}
    input_df = pd.DataFrame(data, index=[timestamp])
    input_df.index.name = "timestamp"
    
    # Extract the hour and day of week from the timestamp
    hour = input_df.index.hour[0]
    dayofweek = input_df.index.dayofweek[0]
    
    # Create dummy variables for hour and day of week
    hour_dummies = pd.get_dummies([hour], prefix="hour")
    day_dummies = pd.get_dummies([dayofweek], prefix="dayofweek")
    
    # Ensure the dummy columns match the expected features.
    # For example, if the model was trained with dummy variables for all 24 hours and 7 days,
    # then we need to add missing columns (set to 0) accordingly.
    expected_hours = [f"hour_{i}" for i in range(24)]
    for col in expected_hours:
        if col not in hour_dummies.columns:
            hour_dummies[col] = 0
    hour_dummies = hour_dummies[expected_hours]
    
    expected_days = [f"dayofweek_{i}" for i in range(7)]
    for col in expected_days:
        if col not in day_dummies.columns:
            day_dummies[col] = 0
    day_dummies = day_dummies[expected_days]
    
    # Combine the dummy features with the continuous features
    # Reset the indices to ensure proper concatenation (row-wise)
    features_df = np.array(pd.concat([hour_dummies.reset_index(drop=True),
                             day_dummies.reset_index(drop=True),
                             input_df.reset_index(drop=True)], axis=1))
    
    # Use the pre-fitted model to predict the energy load
    predicted_energy = model.predict(features_df)
    st.write(f"Predicted Energy Load: {predicted_energy[0].item():.2f}")


