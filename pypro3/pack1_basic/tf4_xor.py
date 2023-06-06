# 논리 게이트 중 XOR 처리
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# model = Sequential([
#     Dense(units=1, input_dim=2),
#     Activation('sigmoid')
# ])
#
# model = Sequential([
#     Dense(units=1, input_dim=2, activation='sigmoid')
# ])

model = Sequential()
# model.add(Dense(units=1, input_dim = 2, activation='sigmoid'))
model.add(Dense(units=5, input_dim = 2, activation='relu'))     # 히든 레이어 안에 있는 활성함수는 relu를 사용
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x, y, epochs=100, batch_size=1, verbose=0)
# batch_size : 몇개의 샘플로 가중치를 낼것인지.
loss_metrics = model.evaluate(x, y)
print("loss_metrics : ", loss_metrics)
pred = model.predict(x > 0.5).astype('int32')
print('예측 결과 : ', pred.flatten())


print('-----'*11)
print(model.input)
print(model.output)
print(model.weights)

print('history : ', history.history)
print('history : ', history.history['loss'])
print('history : ', history.history['accuracy'])

# 학습 시 발생 정보로 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], color = 'red', label='loss')
plt.plot(history.history['accuracy'], color = 'blue', label='accuracy')
plt.xlabel('epochs')
plt.legend(loc='best')
plt.show()