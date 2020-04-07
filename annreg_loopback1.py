from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd 
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from loss import mae, mape, rmse, r_squared, compute_metrics
import os

def base_model():
	model = Sequential()
	model.add(Dense(14, input_dim=3, init='normal', activation='relu'))
	model.add(Dense(8, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='linear'))
	model.compile(loss='mean_squared_error', optimizer = 'adam')
	return model

df = pd.read_csv('Data/6642.csv')
df['time'] = pd.to_datetime(df['time'])
l = len(df)
X = []
for i in range(1, l):
	time = df.loc[i, 'time']
	inten = df.loc[i-1, 'intensity']
	vel = df.loc[i-1, 'velocity']
	#tc = df.loc[i-1, 'tweet_count']
	hour = time.hour
	minute = time.minute
	t = hour * 60 + minute
	X.append([t,vel])
X = np.array(X)
scx = MinMaxScaler()
X = scx.fit_transform(X)

X_train = X[:2000]
X_test = X[2000:]


features = ['intensity', 'velocity']
functions = [rmse, mae, mape, r_squared]
loss = ['rmse', 'mae', 'mape', 'r_squared']


res = pd.DataFrame(columns=['feature', 'rmse', 'mae', 'mape', 'r_squared'])
preds = []
actuals = []
for i, feature in enumerate(features):
	res.loc[i, 'feature'] = feature
	model = base_model()
	y = df[feature].values.reshape(-1, 1)
	y = y[1:]
	scy = MinMaxScaler()
	y = scy.fit_transform(y)
	y_train = y[:2000]
	y_test = y[2000:]
	model.fit(X_train, y_train, batch_size=64, epochs=50)
	y_pred = model.predict(X_test).reshape(-1, 1)
	y_test = scy.inverse_transform(y_test)
	y_pred = scy.inverse_transform(y_pred).astype(int)
	for j, f in enumerate(functions):
		error = f(y_pred, y_test)
		res.loc[i, loss[j]] = error
print(res)
#res.to_csv('Result/ann_6642.csv', index=False)