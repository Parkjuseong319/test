# 다중 선형회귀모델 + Tensorboard
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 

# 5명이 시행한 3회 시험결과로 다음 시험 점수 예측
x_data = np.array([[70,85,80], [71,89,78], [50,80,60], [66,25,60], [50,30,10]])
y_data = np.array([73, 82, 71, 55, 33])

# Sequential api
model = Sequential()
model.add(Dense(6, input_dim=3, activation='linear', name='a'))     # tensorboard 작성시 name 속성값 부여
model.add(Dense(3, activation='linear', name='b'))
model.add(Dense(1, activation='linear', name='c'))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mse'])
history = model.fit(x_data, y_data, batch_size = 1, epochs=30, verbose=2)

# plt.plot(history.history['mse'])
# plt.xlabel('epochs')
# plt.xlabel('loss')
# plt.show()

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data)))    # 0.932

print('--------'*10)

# Functional api
inputs = Input(shape=(3,))
net1 = Dense(6, activation='linear', name='a')(inputs)
net2 = Dense(3, activation='linear', name='b')(net1)
outputs = Dense(1, activation='linear', name='c')(net2)
model2 = Model(inputs, outputs)

model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mse'])
# history = model2.fit(x_data, y_data, batch_size = 1, epochs=30, verbose=2)
# print('설명력 : ', r2_score(y_data, model2.predict(x_data)))    # 0.932

# TensorBoard : 알고리즘에 대한 동작을 시각화해 준다. 시행착오를 최소화 할 수 있음
from keras.callbacks import TensorBoard

tb = TensorBoard(log_dir=".\\my",
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=2,
    embeddings_freq=1,
    embeddings_metadata=None)

history = model2.fit(x_data, y_data, batch_size = 1, epochs=50, verbose=1, callbacks=[tb])
print('설명력 : ', r2_score(y_data, model2.predict(x_data)))

new_data = np.array([[10,10,10], [90,100,95]])
print('예상 점수 : ', model2.predict(new_data).ravel())

# 텐서보드 사용법
# 아나콘다 프롬프트 창에서 python -m tensorboard.main --logdir=my/   를 해주면 도메인 하나 뜨는데 그곳으로 접속시 그래프 볼 수 있다.