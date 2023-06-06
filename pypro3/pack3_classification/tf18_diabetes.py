# 이항분류(sigmoid)는 다항분류(softmax)로 처리할 수 있다.

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

print('sigmoid 사용 ----------------------------------------------')
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
print(x_test[:1])
pred = model.predict([[-0.882353, 0.135678, 0.049180, -0.292929, 0. ,0.00149028, -0.602904, 0.]])
print('예측 결과 : ', pred)
print('예측 결과 : ', np.where(pred > 0.05, 1, 0))

print('다항분류 ---------------------------------------------')
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
model2 = Sequential()
model2.add(Dense(units=16, input_dim=8, activation='relu'))
model2.add(Dense(units=8, activation='relu'))
model2.add(Dense(units=2, activation='softmax'))
print(model2.summary())

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model2.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=2)

loss, acc = model2.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('학습 후의 모델의 정확도 : {:5.2f}'.format(acc * 100))

pred2 = model2.predict([[-0.882353, 0.135678, 0.049180, -0.292929, 0. ,0.00149028, -0.602904, 0.]])
print('예측 결과2 : ', pred2)
print('예측결과 : ', np.argmax(pred2))

