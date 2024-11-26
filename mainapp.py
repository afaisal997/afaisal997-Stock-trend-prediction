import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Data Starting and Ending Point
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter the Ticker For the Asset', 'GOOG')

@st.cache
# Pulling Data from Yahoo Finance
def load_data():
    data = yf.download(user_input, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data()

st.subheader('Data From 2015 Till Date')
st.write(data.describe())

# Charts
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(data.Close, 'c')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig)

scaler = MinMaxScaler(feature_range=(0, 1))

# Loading model
model = load_model('final_model.h5')

data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70):int(len(data))])

# Debugging: Check the type and length of data_testing
print("Type of data_testing:", type(data_testing))
print("Length of data_testing:", len(data_testing))

# Ensure data_testing has at least 100 rows before attempting slicing
if len(data_testing) >= 100:
    past_data_100days = data_testing.tail(100)
    final_data = past_data_100days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_data)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    predicted_value = model.predict(x_test)

    sc = scaler.scale_
    scale_factor = 1 / sc[0]
    predicted_value = predicted_value * scale_factor
    y_test = y_test * scale_factor

    st.subheader('Predictions Vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(predicted_value, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
else:
    st.write("Insufficient data for prediction.")
