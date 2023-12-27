#This is a project that predicts the future price of S&P500


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Load data
company = 'FB'

start = dt.datetime(2012,1,1)
end = dt.datetime(2023,12,12)

data = web.DataReader(company, 'yahoo', start, end)

#Prepare data for neural networks
#Scale down all values to fit in between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

#We are only interested in the price after the markets have closed
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

#Define how many days into the past we need to look at
prediction_days = 90

#define 2 empty lists
x_train = []
y_train = []