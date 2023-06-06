# 피마 당뇨병 데이터로 분류 모델
# Pregnancies: 임신 횟수
# Glucose: 포도당 부하 검사 수치
# BloodPressure: 혈압(mm Hg)
# SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
# Insulin: 혈청 인슐린(mu U/ml)
# BMI: 체질량지수(체중(kg)/(키(m))^2)
# DiabetesPedigreeFunction: 당뇨 내력 가중치 값
# Age: 나이
# Outcome: 클래스 결정 값(0또는 1)

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)

dataset = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/diabetes.csv', delimiter=',')
print(dataset.shape)
print(dataset[:1])

x_train, x_test, y_train, y_test = train_test_split(dataset[:,0:8], dataset[:,-1], test_size= 0.3, random_state=12)

# create model
model = Sequential()
model.add(Dense(units=16, input_dim=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=2)

loss, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('학습 후의 모델의 정확도 : {:5.2f}'.format(acc * 100))

# 시각화
epoch_len = np.arange(len(history.epoch))

plt.plot(epoch_len, history.history['val_loss'], c='red', label='val_loss')
plt.plot(epoch_len, history.history['loss'], c='blue', label="loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

print("********" * 15)
# functional api
from keras.models import Model
from keras.layers import Input

inputs = Input(shape=(8,))
net1 = Dense(units=16, activation='relu')(inputs)
net2 = Dense(units=8, activation='relu')(net1)
outputs = Dense(units=1, activation='sigmoid')(net2)

model2 = Model(inputs, outputs)

print(model2.summary())

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model2.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=2)

loss, acc = model2.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('학습 후의 모델의 정확도 : {:5.2f}'.format(acc * 100))

# 시각화
epoch_len = np.arange(len(history.epoch))

plt.plot(epoch_len, history.history['val_loss'], c='red', label='val_loss')
plt.plot(epoch_len, history.history['loss'], c='blue', label="loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

















