# zoo dataset으로 동물의 type 분류 모델
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

xy = np.loadtxt('../testdata/zoo.csv', delimiter=',')
print(xy[0], xy.shape)  # (101, 17) last column = type

x_data = xy[:, 0:-1]
y_data = xy[:, -1]
print(x_data[0])
print(y_data[0])

# train / test split 생략
print(set(y_data))      # {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

nb_classes = 7
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print(y_one_hot[0])

model = Sequential()
model.add(Dense(32, input_shape=(16,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_data, y_one_hot, epochs=100, batch_size=10, validation_split=0.3, verbose=0)

print(model.evaluate(x_data, y_one_hot))

# 학습도중 발생한 loss, acc를 사용 
his = history.history
loss = his['loss']
val_loss = his['val_loss']
acc = his['acc']
val_acc = his['val_acc']

import matplotlib.pyplot as plt

plt.plot(loss, 'b-', label='trian loss')
plt.plot(val_loss, 'r--', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(acc, 'b-', label='trian accuracy')
plt.plot(val_acc, 'r--', label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# predict
pred_one = x_data[:1]
pred = np.argmax(model.predict(pred_one))
print('예측 - ', pred)
print()
pred_many = x_data[:5]
preds = [np.argmax(i) for i in model.predict(pred_many)]
print('예측 - ', preds)
print('real : ', y_data[:5])