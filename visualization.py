import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import datetime
import streamlit as st

# Aggregate Visulizations of the Clusters:
def timestampcombine(date,time):
    pydatetime = datetime.combine(date, time)
    return pydatetime

def ClusterUnstacker(df):
    df = df.unstack().reset_index() # code 1
    df['timestampstring'] = pd.to_datetime(df.Date.astype("str") + " " + df.level_2.astype("str")) # code 2
    #pd.to_datetime(df.Date  df.level_2) #map(timestampcombine, )
    df = df.dropna()
    return df

def DayvsClusterMaker(df):
    df.index = df.timestampstring
    df['Weekday'] = df.index.map(lambda t: t.date().weekday())
    df['Date'] = df.index.map(lambda t: t.date())
    df['Time'] = df.index.map(lambda t: t.time())
    # Convert 'Date' column to datetime objects before resampling
    df['Date'] = pd.to_datetime(df['Date'])
    DayVsCluster = df.resample('D').mean(numeric_only=True).reset_index(drop=True)
    DayVsCluster = pd.pivot_table(DayVsCluster, values=0, index='ClusterValue', columns='Weekday', aggfunc='count')
    DayVsCluster.columns = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
    return DayVsCluster.T

def visualization(dfcluster_merged):
    if dfcluster_merged is not None:
        st.header("Visualizations of Data to Reveal Patterns of Behaviour over a Time Period")
        st.write(f"Setting Multilevel Indexing for Visualization. The dataframe below shows hierarchical division of the dtaframe- depending on the cluster values assigned by the K means algorithm:")
        dfcluster_merged_viz = dfcluster_merged.set_index(['ClusterValue', 'Date']).T.sort_index() # sets the dataframe with multilevel indexing, 'ClusterValue' as level 0
        st.write(f"Dataframe with multilevel indexing, 'ClusterValue' as level 0:")
        st.write(dfcluster_merged_viz)

        st.header(f"Visualization:")
        st.write(f"Extracted list of unique cluster numbers:")
        clusterlist = list(dfcluster_merged_viz.columns.get_level_values(0).unique())
        st.write(clusterlist)

        st.subheader(f'Plot of the daily energy consumption profiles grouped by clusters:')
        st.write(f"To visualize the cluster patterns, we first look at all the profiles at once grouped by cluster. We  iterate over each cluster to plot the daily energy consumption profiles. The x-axis represents the time of day, and the y-axis represents the total daily profile:")
        
        matplotlib.rcParams['figure.figsize'] = 20, 7
        styles2 = ['LightSkyBlue', 'b','LightGreen', 'g','LightCoral','r','SandyBrown','Orange','Plum','Purple','Gold','b']
        fig, ax = plt.subplots() # Creates a figure and an axes object for plotting
        for col, style in zip(clusterlist, styles2): # Iterates over each cluster number and its corresponding style.
            dfcluster_merged_viz[col].plot(ax=ax, legend=False, style=style, alpha=0.1, xticks=np.arange(0, 86400, 10800)) # Plots the daily profiles for the current cluster number with the specified style.

        ax.set_ylabel('Total Daily Profile')
        ax.set_xlabel('Time of Day')
        # Display the plot in the Streamlit app
        st.pyplot(fig)

        # Unstacking the Dataframe
        st.write(f"Having observed how the energy consumption has been clustered, there is no need for the hierarchy. The multilevel indexing is hence removed from dataframe:")
        dfclusterunstacked = ClusterUnstacker(dfcluster_merged_viz)
        st.write(dfclusterunstacked)
        st.write(f'The following pivot table is created from the unstacked dataframe above:')
        dfclusterunstackedpivoted = pd.pivot_table(dfclusterunstacked, values=0, index='timestampstring', columns='ClusterValue')
        st.write(dfclusterunstackedpivoted)

        st.subheader(f"Total Daily Load Profile Patterns")
        st.write(f"The plot below shows how the various load profiles have been clustered across the year to identify behaviour patterns of total Energy consumption per cluster through time.")
        fig2, ax2 = plt.subplots()
        clusteravgplot = dfclusterunstackedpivoted.resample('D').sum().replace(0, np.nan).plot(ax=ax2, style="^",markersize=15)
        clusteravgplot.set_ylabel('Daily Totals kWh')
        clusteravgplot.set_xlabel('Date')
        # Display the plot in the Streamlit app
        st.pyplot(fig2)

        st.subheader(f"Average Daily Load Profile Patterns")
        st.write(f"The plot below displays the average Energy Consumption in a 24 hour period per cluster.")
        fig3, ax3 = plt.subplots()
        dfclusterunstackedpivoted['Time'] = dfclusterunstackedpivoted.index.map(lambda t: t.time())
        dailyprofile = dfclusterunstackedpivoted.groupby('Time').mean().plot(ax = ax3, figsize=(20,7),linewidth=3, xticks=np.arange(0, 86400, 10800))
        dailyprofile.set_ylabel('Average Daily Profile kWh')
        dailyprofile.set_xlabel('Time of Day')
        dailyprofile.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
        # Display the plot in the Streamlit app
        st.pyplot(fig3)

        st.subheader(f"Plot to show the Clusters Observed (Load) per day")
        st.write(f"What can we deduce from the clusters observed  on average for every day of the Week")
        fig4, ax4 = plt.subplots()
        DayVsCluster = DayvsClusterMaker(dfclusterunstacked)
        DayVsClusterplot1 = DayVsCluster.plot(ax = ax4, figsize=(20,7),kind='bar',stacked=True)
        DayVsClusterplot1.set_ylabel('Number of Days in Each Cluster')
        DayVsClusterplot1.set_xlabel('Day of the Week')
        DayVsClusterplot1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
        # Display the plot in the Streamlit app
        st.pyplot(fig4)

        st.subheader(f"5th Plot shows the how much capacity each cluster (of Energy load) occupies per day")
        st.write(f"The plot below puts more perspective to the plot #4 above.")
        fig5, ax5 = plt.subplots()
        DayVsClusterplot2 = DayVsCluster.T.plot(ax = ax5, figsize=(20,7),kind='bar',stacked=True, color=['b','g','r','c','m','y','k']) #, color=colors2
        DayVsClusterplot2.set_ylabel('Number of Days in Each Cluster')
        DayVsClusterplot2.set_xlabel('Cluster Number')
        DayVsClusterplot2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Display the plot in the Streamlit app
        st.pyplot(fig5) 


def plot_eda_dataframe(eda_df):
    """
    Plots a line chart of the provided DataFrame and displays it in a Streamlit app.

    Parameters:
    - eda_df (pd.DataFrame): A DataFrame with a datetime index and columns to plot.
    """
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate the line plot using the DataFrame's built-in plot function
    eda_df.plot(ax=ax)
    
    # Customize the plot as needed
    ax.set_title("Weather Data Pattern Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Humidity, Temperature & Dew Point Values")
    
    # Display the plot in the Streamlit app
    st.pyplot(fig)


def plot_energy_data(df_prediction_data):
    """
    Plots a line chart of the energy prediction data and displays it in a Streamlit app.

    Parameters:
    - df_prediction_data (pd.DataFrame): A DataFrame containing energy prediction data with a datetime index.
    """
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate the line plot using the DataFrame's built-in plot function
    df_prediction_data.plot(ax=ax)
    
    # Customize the plot
    ax.set_title("Energy Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Consumption")
    
    # Display the plot in the Streamlit app
    st.pyplot(fig)


def plot_correlation_matrix(df):
    """
    Computes the correlation matrix of the given DataFrame and displays it as a heatmap in a Streamlit app.

    Parameters:
    - df (pd.DataFrame): DataFrame with numerical columns for which the correlation matrix is computed.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a heatmap of the correlation matrix using seaborn
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    
    # Set a title for the plot
    ax.set_title("Correlation Matrix")
    
    # Display the heatmap in the Streamlit app
    st.pyplot(fig)


def plot_energy_scatter_plots(combined_data):
    """
    Creates a figure with a 1x3 grid of scatter plots:
        - TemperatureC vs energy_load
        - Humidity vs energy_load
        - Dew PointC vs energy_load

    Parameters:
    -----------
    combined_data : pd.DataFrame
        DataFrame that contains the following columns:
            - 'TemperatureC'
            - 'Humidity'
            - 'Dew PointC'
            - 'Energy_load'
    
    Returns:
    --------
    None
        The function displays the plot using st.pyplot().
    """
    # Create a figure and a 1x3 grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot 1: TemperatureC vs energy_load
    axs[0].scatter(combined_data['TemperatureC'], combined_data['Energy_Load'])
    axs[0].set_xlabel("TemperatureC")
    axs[0].set_ylabel("Energy Load")
    axs[0].set_title("TemperatureC vs Energy Load")
    
    # Scatter plot 2: Humidity vs energy_load
    axs[1].scatter(combined_data['Humidity'], combined_data['Energy_Load'])
    axs[1].set_xlabel("Humidity")
    axs[1].set_ylabel("Energy Load")
    axs[1].set_title("Humidity vs Energy Load")
    
    # Scatter plot 3: Dew PointC vs energy_load
    axs[2].scatter(combined_data['Dew PointC'], combined_data['Energy_Load'])
    axs[2].set_xlabel("Dew PointC")
    axs[2].set_ylabel("Energy Load")
    axs[2].set_title("Dew PointC vs Energy Load")
    
    # Adjust layout and display the plots
    plt.tight_layout()
    st.pyplot(fig)


def plot_predicted_vs_actual(predicted_vs_actual):
    """
    Creates a scatter plot comparing actual and predicted energy consumption.
    
    Parameters:
    -----------
    predicted_vs_actual : pd.DataFrame
        A DataFrame containing two columns:
            - 'Actual': The actual energy consumption values.
            - 'Predicted': The predicted energy consumption values.
    
    Returns:
    --------
    None
        The function displays the scatter plot.
    """
    # Create a new figure and axis object
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the scatter plot
    ax.scatter(predicted_vs_actual['Actual'], predicted_vs_actual['Predicted'], 
               color='blue', alpha=0.7)
    
    # Label the axes
    ax.set_xlabel("Actual Energy Consumption")
    ax.set_ylabel("Predicted Energy Consumption")
    
    # Optionally add a title and grid for better readability
    ax.set_title("Actual vs Predicted Energy Consumption")
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

def lplot_predicted_vs_actual(predicted_vs_actual):
    """
    Plots a line plot of predicted values against actual values.

    Parameters:
    -----------
    predicted_vs_actual : pd.DataFrame
        A DataFrame containing two columns:
            - 'Actual': The actual energy consumption values.
            - 'Predicted': The predicted energy consumption values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predicted_vs_actual['Actual'], label='Actual Values', color='blue', linestyle='-', marker='o')
    ax.plot(predicted_vs_actual['Predicted'], label='Predicted Values', color='red', linestyle='--', marker='x')
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.set_title('Predicted vs Actual Values')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


