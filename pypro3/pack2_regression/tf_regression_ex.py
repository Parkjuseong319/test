# 문제1)
# http://www.randomservices.org/random/data/Galton.txt
# data를 이용해 아버지 키로 아들의 키를 예측하는 회귀분석 모델을 작성하시오.
# train / test 분리
# Sequential api와 function api 를 사용해 모델을 만들어 보시오.
# train과 test의 mse를 시각화 하시오
# 새로운 아버지 키에 대한 자료로 아들의 키를 예측하시오.

# father  mother  ...  childNum  gender  childHeight

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, Layer, Dense
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('../testdata/galton.csv')
print(data.head(3))
data = data.loc[data['gender'] == 'male']
x = data['father']
y = data['childHeight'].values
# print(x[:3])
# print(y[:3])
print(np.corrcoef(x, y))


# train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# create model
model1 = Sequential()
model1.add(Dense(units=2, input_dim=1, activation='linear'))
model1.add(Dense(units=5, activation='linear'))
model1.add(Dense(units=1, activation='linear'))
print(model1.summary())


opti = optimizers.Adam(learning_rate= 0.001)
model1.compile(optimizer=opti, loss='mse', metrics=['mse'])
hitory = model1.fit(x=x_train, y=y_train, batch_size=64, epochs=1000, verbose=0)

loss_metrics = model1.evaluate(x=x_test, y=y_test)
print('loss_metrics1 : ', loss_metrics)
print("설명력1 : ", r2_score(y_test, model1.predict(x_test)))
print('실제값1 : ', y_test)
print('예측값1 : ', model1.predict(x_test).flatten())
# print('분류 정확도1 : ', accuracy_score(y_test, model1.predict(x_test)))

# model2
inputs = Input(shape=(1,))      
output1 = Dense(units=2, activation='linear')(inputs)      
output2 = Dense(units=5, activation='linear')(output1)
output3 = Dense(units=1, activation='linear')(output2)
model2 = Model(inputs, output3)
print(model2.summary())

model2.compile(optimizer=opti, loss='mse', metrics=['mse'])
hitory2 = model2.fit(x=x_train, y=y_train, batch_size=64, epochs=1000, verbose=0)
loss_metrics = model2.evaluate(x=x_test, y=y_test)
print('loss_metrics2 : ', loss_metrics)

print("설명력2 : ", r2_score(y_test, model2.predict(x_test)))
print('실제값2 : ', y_test)
print('예측값2 : ', model2.predict(x_test).flatten())
# print('분류 정확도2 : ', accuracy_score(y_test, model2.predict(x_test)))


# 시각화 
plt.rc('font', family='malgun gothic')
plt.plot(hitory.history['mse'], label='Sequential', color='r')
plt.plot(hitory2.history['mse'], label='functional', color='b')
plt.legend()
plt.show()

# 새로운 값으로 예측하기
new_data = new_data = [79.5, 80.3, 81.8]
new_data = np.expand_dims(new_data, axis=1)
print('새로운 예측값 : ', model1.predict(new_data).flatten())