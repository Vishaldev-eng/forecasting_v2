import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyodbc
import os
import pandas as pd

# Streamlit page config
st.set_page_config(page_title='Forecasting', layout='centered')

st.title("Forecasting Application")

st.write("""
Easily connect to your Server, import data by writing SQL queries, and generate future forecasts.
         
1️⃣ Enter DB connection details  
2️⃣ Select a database  
3️⃣ Write and run your SQL query  
4️⃣ Select date and target columns for forecasting  
5️⃣ Run the forecast and view interactive plots  
""")

# Input DB connection credentials
server = st.text_input("Enter DB Server", value=os.getenv('DB_SERVER', ''))
user = st.text_input("Enter DB User", value=os.getenv('DB_USER', ''))
password = st.text_input("Enter DB Password", type="password", value=os.getenv('DB_PASSWORD', ''))
initial_database = st.text_input("Enter Initial DB Name", value=os.getenv('DB_NAME', ''))

# Session state to store connection
if "connection_established" not in st.session_state:
    st.session_state.connection_established = False
if "server_connection" not in st.session_state:
    st.session_state.server_connection = None

if st.button("Connect"):
    try:
        conn_str_initial = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={initial_database};UID={user};PWD={password}"
        connection_initial = pyodbc.connect(conn_str_initial)
        st.session_state.server_connection = connection_initial
        st.session_state.connection_established = True
        st.success("Connected successfully!")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.session_state.connection_established = False

# After successful connection
if st.session_state.connection_established:

    connection_initial = st.session_state.server_connection

    try:
        db_query = "SELECT name FROM sys.databases"
        dbs = pd.read_sql(db_query, connection_initial)
        db_list = dbs['name'].tolist()

        selected_db = st.selectbox("Select Database", db_list)

        if selected_db:
            conn_str_db = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={selected_db};UID={user};PWD={password}"
            connection_db = pyodbc.connect(conn_str_db)
            st.write(f"Connected to database: **{selected_db}**")

            sql_query = st.text_area("Enter your SQL query", height=200)

            if st.button("Import Data"):
                try:
                    df = pd.read_sql_query(sql_query, connection_db)
                    st.session_state['df'] = df
                    st.session_state.query_executed = True
                    st.success("Data imported successfully.")
                except Exception as e:
                    st.error(f"Failed to import data: {e}")
                    st.session_state.query_executed = False

            if "query_executed" in st.session_state and st.session_state.query_executed:

                df = st.session_state['df']
                st.subheader("Imported Data")
                st.dataframe(df)

                date_column = st.selectbox("Select Date Column", df.columns)
                value_column = st.selectbox("Select Target Column", df.columns)

                freq_options = {
                    'D: calendar day': 'D',
                    'W: weekly': 'W',
                    'MS: month start': 'MS',
                    'ME: month end': 'ME',
                    'QS: quarter start': 'QS',
                    'QE: quarter end': 'QE',
                    'YS: year start': 'YS',
                    'YE: year end': 'YE'
                }
                freq_label = st.selectbox("Select Frequency", list(freq_options.keys()))
                freq_input = freq_options[freq_label]

                periods_input = st.number_input('Forecast Periods into Future:', min_value=1, max_value=500, value=100)

                model_choice = st.radio("Choose Forecasting Model", ("Prophet", "Exponential Smoothing", "SARIMA", "Moving Average"))

                if model_choice == "Moving Average":
                    ma_window = st.slider("Moving Average Window (periods)", min_value=2, max_value=24, value=3)
                else:
                    ma_window = None

                if st.button("Run Forecast"):
                    df_subset = df[[date_column, value_column]].copy()
                    df_subset.rename(columns={date_column: 'ds', value_column: 'y'}, inplace=True)
                    df_subset['ds'] = pd.to_datetime(df_subset['ds'])
                    df_subset['y'] = pd.to_numeric(df_subset['y'], errors='coerce')
                    df_subset.dropna(subset=['ds', 'y'], inplace=True)
                    df_subset.sort_values('ds', inplace=True)

                    st.subheader("Input Data for Forecasting")
                    st.dataframe(df_subset)

                    if model_choice == "Prophet":
                        try:
                            model = Prophet()
                            model.fit(df_subset)
                            future = model.make_future_dataframe(periods=periods_input, freq=freq_input)
                            forecast = model.predict(future)

                            st.subheader("Forecast Results")
                            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input))

                            st.subheader("Forecast Plot")
                            st.plotly_chart(plot_plotly(model, forecast))

                            st.subheader("Forecast Components")
                            st.plotly_chart(plot_components_plotly(model, forecast))
                        except Exception as e:
                            st.error(f"Prophet Forecast failed: {e}")

                    elif model_choice == "Exponential Smoothing":
                        df_subset.set_index('ds', inplace=True)
                        ts_data = df_subset['y']
                        model = ExponentialSmoothing(ts_data, trend='add', seasonal='add')
                        model_fit = model.fit()
                        forecast = model_fit.forecast(periods_input)

                        forecast_df = forecast.reset_index()
                        forecast_df.columns = ['date', 'forecast']

                        st.subheader("Forecasted Data")
                        st.dataframe(forecast_df)

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(ts_data, label="Original Data")
                        ax.plot(model_fit.fittedvalues, label="Fitted")
                        ax.plot(forecast, label="Forecast")
                        ax.set_title("Triple Exponential Smoothing Forecast")
                        ax.legend()
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=90)
                        ax.set_xticks(list(ts_data.index) + list(forecast.index))
                        st.pyplot(fig)

                    elif model_choice == "SARIMA":
                        df_subset.set_index('ds', inplace=True)
                        ts_data = df_subset['y']
                        model = SARIMAX(ts_data, order=(0, 1, 1), seasonal_order=(2, 1, 1, 4))
                        model_fit = model.fit()
                        forecast = model_fit.predict(start=len(ts_data), end=len(ts_data) + periods_input - 1)
                        forecast_df = forecast.reset_index()
                        forecast_df.columns = ['date', 'forecast']

                        st.subheader("Forecasted Data")
                        st.dataframe(forecast_df)

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(ts_data, label="Original Data")
                        ax.plot(forecast, label="SARIMA Forecast")
                        ax.set_title("SARIMA Forecast")
                        ax.legend()
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=90)
                        ax.set_xticks(list(ts_data.index) + list(forecast.index))
                        st.pyplot(fig)

                    elif model_choice == "Moving Average":
                        df_subset.set_index('ds', inplace=True)
                        ts_data = df_subset['y']
                        rolling_mean = ts_data.rolling(window=ma_window).mean()
                        last_avg = rolling_mean.dropna().iloc[-1]
                        forecast = pd.Series(
                            [last_avg] * periods_input,
                            index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(1),
                                                periods=periods_input, freq=freq_input)
                        )

                        forecast_df = forecast.reset_index()
                        forecast_df.columns = ['date', 'forecast']

                        st.subheader("Forecasted Data")
                        st.dataframe(forecast_df)

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(ts_data, label="Original Data")
                        ax.plot(rolling_mean, label=f"{ma_window}-Period Moving Average", linestyle='--')
                        ax.plot(forecast, label="Forecast")
                        ax.set_title(f"Moving Average Forecast ({ma_window} periods)")
                        ax.legend()
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=90)
                        ax.set_xticks(list(ts_data.index) + list(forecast.index))
                        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to fetch database list: {e}")
