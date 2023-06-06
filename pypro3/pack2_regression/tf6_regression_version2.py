# 단순 선형회귀 모델 작성 - keras library(module) 사용
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

x_data = [1.,2.,3.,4.,5.]
y_data = [1.2,2.0,3.0,3.5,5.5]
print('상관계수 : ', np.corrcoef(x_data, y_data))       #  0.9749

# 단순선형회귀 모델 작성을 위한 network를 구성
model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
# mse - mean square error. 평균 제곱오차
model.fit(x_data, y_data, batch_size=1, epochs=100, verbose=1)
print('evaluate : ', model.evaluate(x_data, y_data))    # loss 값 출력

pred = model.predict(x_data)
print('예측값', pred.flatten())

import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, pred, 'b')
plt.show()

# 결정계수
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, pred))

# 새로운 값으로 예측
new_x = [1.5, 2.5, 3.5]
print('예측 결과 : ', model.predict(new_x).flatten())


