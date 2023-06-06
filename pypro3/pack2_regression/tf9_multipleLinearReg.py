# 다중선형회귀 : feature 간 단위의 차이가 클 때 정규화/표준화 작업이 효과적
# 정규화/표준화 작업 대상은 feature
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
import pandas as pd
import numpy as np
# StandardScaler : 기본 스케일러. 평균과 표준편차를 사용. 이상치가 있으면 불균형해진다.
# MinMaxScaler : 최대, 최소값이 0, 1이 되도록 정규화. 이상치에 민감
# RobustScaler : 이상치의 영향을 최소화함. 중앙값과 사분위수를 사용


data = pd.read_csv('../testdata/Advertising.csv')
del data['no']
print(data.head(3))

fdata = data[['tv', 'radio', 'newspaper']]
ldata = data.iloc[:, [3]]

print(fdata.head(2))
print(ldata.head(2))

np.random.seed(123)
# 정규화 방법 1
scaler = MinMaxScaler(feature_range=(0,1))
fedata = scaler.fit_transform(fdata)
print(fedata[:1])

# 정규화 방법 2
# fedata = minmax_scale(fdata, axis=0, copy=True)
# print(fedata[:1])

# train / test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fedata, ldata, shuffle=True, test_size=0.3, random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Dense(units=20, input_dim=3, activation='linear'))   # units는 보통 2의 배수를 준다. hidden layer에서 'relu'도 가능
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary())

import tensorflow as tf
tf.keras.utils.plot_model(model, 'abc.png')     # graphviz를 설치해야 정상적으로 작동된다.

# 학습 도중 조기 종료 가능
# 학습의 조기종료는 validation에 의해 조기종료된다.
from keras.callbacks import EarlyStopping 
es = EarlyStopping(patience=3, mode='auto', monitor='val_loss')      # patience는 보통 10을 준다.

history = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])
# validation_split 값을 줘야 val_loss가 생성된다. 

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])

# history 값
print('history : ', history.history)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
# val_loss(validation)의 그래프와 loss(train)그래프의 간격이 점점 벌어진다면 overfitting이다.

from sklearn.metrics import r2_score
print('r2_score', r2_score(y_test, model.predict(x_test)))

