import streamlit as st
import pickle
import numpy as np
import datetime
import pandas as pd
import math
from math import log
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from visualization import plot_eda_dataframe, plot_energy_data, plot_correlation_matrix, plot_energy_scatter_plots, plot_predicted_vs_actual, lplot_predicted_vs_actual

def load_model():
    with open('kmeans_centers.pkl', 'rb') as file:
        centers = pickle.load(file)
    return centers

#  Calculate Dew Point
def calculate_dewpoint(tempC, humidity):
    """
    Calculate the dew point in Celsius given the temperature (tempC) in Celsius and relative humidity (humidity) in percent.
    
    Uses the Magnus-Tetens approximation:
        dew_point = (b * alpha) / (a - alpha)
    where:
        alpha = (a * tempC) / (b + tempC) + ln(RH/100)
        a = 17.27, b = 237.7°C
    
    Parameters:
        tempC (float): Temperature in Celsius.
        humidity (float): Relative humidity in percent.
    
    Returns:
        float: Calculated dew point in Celsius.
    """
    a = 17.27
    b = 237.7
    alpha = (a * tempC) / (b + tempC) + np.log(humidity / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

with open ('knn_model.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    columns_knn = data['columns']



import pandas as pd
import numpy as np

def create_feature_array(energy_data, eda_df):
    """
    Generates a feature array by one-hot encoding the hour and day of the week
    from the 'energy_data' index and concatenating these with the 'eda_df' DataFrame.
    Aligns the resulting DataFrame to match the number of rows in 'energy_data' and fills missing values with column means.

    Parameters:
    -----------
    energy_data : pd.DataFrame
        DataFrame with a DatetimeIndex containing energy consumption data.
    eda_df : pd.DataFrame
        DataFrame containing additional features to be concatenated.

    Returns:
    --------
    np.ndarray
        A NumPy array containing the combined features, aligned to the number of rows in 'energy_data' and with missing values filled.
    """
    # Ensure 'energy_data' index is a DatetimeIndex
    if not isinstance(energy_data.index, pd.DatetimeIndex):
        raise TypeError("The index of 'energy_data' must be a DatetimeIndex.")

    # One-hot encode the hour and day of the week
    hour_dummies = pd.get_dummies(energy_data.index.hour, prefix='hour')
    dayofweek_dummies = pd.get_dummies(energy_data.index.dayofweek, prefix='dayofweek')

    # Reset index of 'eda_df' and fill missing values with column means
    eda_df_filled = eda_df.reset_index(drop=True).fillna(eda_df.mean())

    # Ensure the number of rows match before concatenation
    min_length = min(len(hour_dummies), len(dayofweek_dummies), len(eda_df_filled))
    hour_dummies = hour_dummies.iloc[:min_length]
    dayofweek_dummies = dayofweek_dummies.iloc[:min_length]
    eda_df_filled = eda_df_filled.iloc[:min_length]

    # Concatenate the DataFrames
    combined_df = pd.concat([hour_dummies, dayofweek_dummies, eda_df_filled], axis=1)

    # Align the combined DataFrame to match the number of rows in 'energy_data'
    current_rows = combined_df.shape[0]
    target_rows = len(energy_data)

    if current_rows < target_rows:
        # Add dummy rows with NaN values
        additional_rows = target_rows - current_rows
        dummy_rows = pd.DataFrame(np.nan, index=range(additional_rows), columns=combined_df.columns)
        combined_df = pd.concat([combined_df, dummy_rows], ignore_index=True)
    elif current_rows > target_rows:
        # Remove excess rows
        combined_df = combined_df.iloc[:target_rows]

    # Fill any remaining NaN values with column means
    combined_df = combined_df.fillna(combined_df.mean())

    # Convert to a NumPy array
    features = combined_df.to_numpy()

    return features

def show_explore_page():
    st.header("KNN Model Prediction of Metadata")
    
    # Allow the user to upload a CSV file
    st.write(f'Upload the weather dataset:')
    file_weather = st.file_uploader("Choose a CSV file", type="csv", key='low')
    st.write(f'Upload the energy consumption dataset:')
    file_energy = st.file_uploader("Upload Energy File", type = "csv", key = 'high')
    energy_data = pd.read_csv(file_energy, index_col = "timestamp", parse_dates=True)

    # st.subheader('Click below to get energy load and cluster value information')
    if file_weather is not None:
        try:
            # Read the uploaded file into a DataFrame
            weather_data = pd.read_csv(file_weather, index_col = "timestamp", parse_dates=True)
            st.write("The Uploaded weather data Dataset:")
            required_columns = {'TemperatureC', 'Humidity', 'Dew PointC'}
            st.write(weather_data)

            if not required_columns.issubset(weather_data.columns):
                st.error("The uploaded file must contain 'timestamp', 'TemperatureC', 'Humidity', and 'Dew PointC' columns.")
                return
            
            # Resample Dataframe to hourly data values and remove outliers. 
            weather_hourly = weather_data.resample("h").mean(numeric_only=True)
            weather_hourly_nooutlier = weather_hourly[weather_hourly > -40]
            weather_hourly_nooutlier_nogaps = weather_hourly_nooutlier.fillna(method='ffill')

            # Inspect the available columns in the dataframe
            st.write(f"Available columns in the weather dataset:")
            st.write(weather_hourly_nooutlier_nogaps.columns)

            # Take the Columns we need for Prediction
            eda_df = weather_hourly_nooutlier_nogaps[['TemperatureC', 'Humidity', 'Dew PointC']]
            st.write(f'Weather data features under consideration:')
            st.write(eda_df)  # Display a preview of the DataFrame

            st.write(f"Energy consumption pattern over time")
            plot_energy_data(energy_data)
            st.write(f'Plot showing change in Humidity, Temperature and DewPoint readings over time:')
            plot_eda_dataframe(eda_df)

            corr_wdf = eda_df.reset_index(drop=True)

            # Reset the index of 'df_prediction_data' to ensure unique index values
            energy_data_c = energy_data.reset_index(drop=True)
            combined_data = pd.concat([corr_wdf, energy_data_c], axis=1)
            combined_data = pd.DataFrame(combined_data)
            # Rename the last column to 'Energy_Load'
            combined_data.rename(columns={combined_data.columns[-1]: "Energy_Load"}, inplace=True)
            st.write(f"Dataframe showing concanated Weather and Energy Data:")
            st.write(combined_data)
            st.write(f"Size of the combined dataframe: {combined_data.shape}")

            st.write(f"Correlation Matrix:")
            plot_correlation_matrix(combined_data)

            st.write(f"Scatter plots showing the relationship between weather data features and energy values:")
            plot_energy_scatter_plots(combined_data)

            features = create_feature_array(energy_data, eda_df)

            prediction = model.predict(features)

            # merging the testdata dataframe with a dataframe called predictions created from the array above:
            predicted_vs_actual = pd.concat([energy_data, pd.DataFrame(prediction, index=energy_data.index)], axis=1)
            predicted_vs_actual.columns = ['Actual', 'Predicted']
            st.write(f"DataFrame showing Predicted and Actual Data")
            st.write(predicted_vs_actual)
            st.write(f"Scatter plot to show relationship between KNN model's predicted data and the actual data")
            plot_predicted_vs_actual(predicted_vs_actual)

            st.write(f"Plot showing the predicted energy consumption data vs actual energy consumption patterns in the test period:")
            lplot_predicted_vs_actual(predicted_vs_actual)

            st.subheader(f"Evaluation Metrics:")
            errors = abs(predicted_vs_actual['Predicted'] - predicted_vs_actual['Actual'])
            # Calculate mean absolute percentage error (MAPE) and add to list
            MAPE = 100 * np.mean((errors / predicted_vs_actual['Actual']))
            st.write(f'Mean Absolute Percentage error (MAPE) of the data set: {MAPE:.2f} %')

            # Mean Squared Error of the Cross Validation Set:
            mse = mean_squared_error(predicted_vs_actual['Actual'], predicted_vs_actual['Predicted']) 
            st.write(f"mean squared error (MSE) of the Predicted Data: {mse:.2f}")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")










