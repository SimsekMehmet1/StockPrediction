#This is a project that predicts the future sotck price


import pandas as pd   #Used for data manipulation
import numpy as np		#Used for numerical operations
import matplotlib.pyplot as plt  #Used for plotting data
import pandas_datareader as web		#Used for fetchng financial data from web
import datetime as dt        #Handles date objects

from sklearn.preprocessing import MinMaxScaler  #Used to scale data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
#Tensorflow.keras is use to build the LSTM neural network model


#Load data
#FB is the stock ticker for facebook
company = 'FB'

#Determine the start and end date of retrieving the data
start = dt.datetime(2012,1,1)
end = dt.datetime(2023,12,12)

#Using datareader from panda_datareader fetch stock prices for facebook from yahoo finance covering the start and end date specified above
data = web.DataReader(company, 'yahoo', start, end)

#Prepare data for neural networks
#Scale down all values to fit in between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

#We are only interested in the price after the markets have closed
#This line scales the closing price of the stock
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

#Define how many days into the past we need to look at
prediction_days = 90

#define 2 empty lists
x_train = []
y_train = []

#for loop to fill both trains with sclaed data
for x in range(prediction_Days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])


#Reshape the data for LSTM input
x_train, y_tain = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Start building the model
model = Sequential()

model.add(LSTM(units = 50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
          
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

