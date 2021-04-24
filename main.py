import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM , Dropout , Dense

# Importing Data

dataset_train = pd.read_csv('./Dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[: , 1:2].values

dataset_test = pd.read_csv('./Dataset/Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[: , 1:2].values

dataset_total = pd.concat((dataset_train['Open'] , dataset_train['Open']),axis = 0)
# preprocessing data / Feature scaling

sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# data structure making timestamp 60 and 1 output
X_train = []
y_train = []
for i in range(60 , len(dataset_train)):
    X_train.append(training_set_scaled[i-60:i , 0])
    y_train.append(training_set_scaled[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train ,(X_train.shape[0],X_train.shape[1] , 1))
# Building the model

regressor = Sequential([
    LSTM(units = 50 , return_sequences = True , input_shape = (X_train.shape[1] , 1)),
    Dropout(0.2),
    LSTM(units = 50 , return_sequences = True),
    Dropout(0.2),
    LSTM(units = 50 , return_sequences = True),
    Dropout(0.2),
    LSTM(units = 50),
    Dropout(0.2),
    Dense(1)
])

regressor.compile(optimizer ='adam', loss = 'mean_squared_error')

regressor.fit(X_train , y_train , epochs = 100 , batch_size = 32)

# Predictions
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
scaled_inputs = sc.transform(inputs)

X_test = []
for i in range(60 ,80):
    X_test.append(scaled_inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test ,( X_test.shape[0] , X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = test_set

# Visualising the result
plt.figure(figsize = (12,10))
plt.plot(real_stock_price , color = 'red' , label = 'Real Srock Price')
plt.plot(predicted_stock_price , color = 'red' , label = 'Predicted Srock Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Real vs Predicted Stock Price Using LSTM ')
plt.show()







