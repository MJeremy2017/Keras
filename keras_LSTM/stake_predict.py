import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


dat = pd.read_csv('D:\Sports Demand Prediction\sportsdemand\Demand prediction\datasetFInal\dataset.csv', parse_dates=['Date'])
dat['year'] = dat['Date'].dt.year
dat['month'] = dat['Date'].dt.month

stake_month = dat[['year', 'month', 'Stake']].groupby(['year', 'month'], as_index=False).agg({'Stake': {'Stake': 'sum'}})
stake_month.columns = ['year', 'month', 'stake']

plt.plot(stake_month.stake)
plt.show()

stake_month.shape

# scale data
train = stake_month.stake.values.reshape(len(stake_month), -1)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

plt.plot(train_scaled)
plt.show()


def shift_dat(data, shift=1):
    d_X = []
    d_y = []
    for i in range(data.shape[0]-shift):
        d_X.append(data[i:(i+shift), 0])
        d_y.append(data[(i+shift), 0])
    return np.array(d_X).flatten(), np.array(d_y).flatten()


# shift the data by 1
train, test = train_scaled[:57], train_scaled[57:]

train_X, train_y = shift_dat(train, shift=1)
test_X, test_y = shift_dat(test, shift=1)

train_X = train_X.reshape(len(train_X), 1, 1)
test_X = test_X.reshape(len(test_X), 1, 1)

# build model
model = Sequential()
model.add(LSTM(6, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(train_X, train_y, batch_size=1, epochs=300)

test_pred = model.predict(test_X)
test_pred = scaler.inverse_transform(test_pred)

plt.plot(stake_month.stake)
plt.plot(np.concatenate((stake_month.stake.values[:56], test_pred.flatten())))
plt.show()