# 다중 선형회귀 : 주식데이터 사용 - 전날 데이터로 다음날 종가 예측
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
import numpy as np

xy = np.loadtxt("../testdata/stockdaily.csv", delimiter=',', skiprows=1)
print(xy[:3], xy.shape)     # (732, 5)

# feature 열 정규화
x_data = xy[:, 0:-1]        # Close를 제외한 column들이 feature
print(x_data[:2])
scaler = MinMaxScaler(feature_range=(0, 1))
x_data = scaler.fit_transform(x_data)
# print(scaler.inverse_transform(x_data))
print(x_data[:2])

y_data = xy[:, [-1]]        # label : 주식 종가(Close)
print(y_data[:3])

print()
# 삭제 전 
print(x_data[0], y_data[0])
print(x_data[1], y_data[1])
print()
x_data = np.delete(x_data, -1, axis=0)   # 마지막행 삭제
y_data = np.delete(y_data, 0, axis=0)   # 종가 제일 첫 행 삭제

# 삭제 후
print(x_data[0], y_data[0])

print('--------'*10)
model = Sequential()
model.add(Dense(units=1, input_dim=4, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(x_data, y_data, epochs=200, verbose=0)
print('evaluate : ', model.evaluate(x_data, y_data))

print(x_data[10])
test = x_data[10].reshape(-1, 4)        # 2차원으로 변경
print('실제값 : ', y_data[10])
print('예측값 : ', model.predict(test).flatten())

from sklearn.metrics import r2_score
pred = model.predict(x_data)
print('r2_score : ',r2_score(y_data, pred))    # 0.99386 << overfitting 의심

import matplotlib.pyplot as plt

plt.plot(y_data, 'b', label='real')
plt.plot(pred, 'r--', label='pred')
plt.legend()
plt.show()

print('\n과적합이 의심스러운 경우 train / test split')
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False, random_state=123)
# 시계열 자료이므로 shuffle을 False를 줘서 데이터가 안섞이도록 한다.
print(len(x_data))  # 731
train_size = int(len(x_data) * 0.7)
test_size = int(len(x_data) - train_size)
print(train_size, test_size)    # 511 220
x_train, x_test = x_data[0:train_size], x_data[train_size:len(x_data)]
y_train, y_test = y_data[0:train_size], y_data[train_size:len(x_data)]
print(x_train[:2], x_train.shape)
print(y_train[:2], y_train.shape)

model2 = Sequential()
model2.add(Dense(units=1, input_dim=4, activation='linear'))

model2.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model2.fit(x_train, y_train, epochs=200, verbose=0)
print('split -> evaluate : ', model2.evaluate(x_test, y_test))

pred2 = model2.predict(x_test)
print('split -> r2_score : ',r2_score(y_test, pred2))       # 0.9475

plt.plot(y_test, 'b', label='real')
plt.plot(pred2, 'r--', label='pred')
plt.legend()
plt.show()

# 머신러닝의 이슈 : 최적화와 일반화의 줄다리기!



