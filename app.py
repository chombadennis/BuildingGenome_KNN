import streamlit as st
from predict_page import predict_energy_from_input
from explore_page import show_explore_page


st.title('K- Nearest Neighbor Regression Model')

st.header('Electricity Prediction using Regression for Measurement and Verification')

st.write(
    """
    Prediction is valuable for anomaly detection, load profile-based building control and measurement and verification procedures.
    Below, we predict Energy load of a building from a KNN- model built from historical weather data and the corresponding energy consumption patterns
    """
)

# Sidebar for navigation
page = st.sidebar.selectbox("Select a Page", ("Predict", "Explore"))

if page == "Predict":
    predict_energy_from_input()
elif page == "Explore":
    show_explore_page()

