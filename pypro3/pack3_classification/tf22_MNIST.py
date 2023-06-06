# MNIST dataset으로 이미지 분류 - 다항분류
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

(x_train, y_train), (x_test, y_test) =tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,). 이미지 3차원, label 1차원
print(x_train[0])   # feature
print(y_train[999])   # label

# for i in x_train[999]:
#     for j in i:
#         sys.stdout.write('%s  '%j)
#     sys.stdout.write('\n')
# plt.imshow(x_train[999], cmap='gray')
# plt.show()

x_train = x_train.reshape(60000, 784).astype('float32')     # 28 by 28 => 784(28²)
x_test = x_test.reshape(10000, 784).astype('float32')     # 28 by 28 => 784(28²)
print(x_train[0])

x_train /= 255.0    # 정규화 : 필수는 아니지만 해주면 대개의 경우 성능 향상
x_test /= 255.0
print(x_train[0])

print(set(y_train)) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# label은 one-hot encoding 처리 : 출력층 활성화 함수가 softmax이므로 해줌.
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
print(y_train[0])

# train dataset의 일부를 validation data로 사용
x_val = x_train[50000:60000]
y_val = y_train[50000:60000]
x_train = x_train[0:50000]
y_train = y_train[0:50000]
print(x_val.shape, x_train.shape)       # (10000, 784) (50000, 784)

# model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

model = Sequential()
# 방법 1
"""
model.add(Dense(units=128, input_shape=(784,)))
# reshape를 하지 않은 경우 --------------------
# model.add(Flatten(input_shape=(28, 28)))   
# model.add(Dense(units=128))
# -----------------------------------------
model.add(Activation('relu'))
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
"""

# 방법 2
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))        # 레이어 사이마다 넣어준다.
model.add(Dense(units=10, activation='softmax'))
# Overfitting 방지 방법
# Drop-out은 어떤 특정한 설명변수 Feature만을 과도하게 집중하여 학습함으로써 발생할 수 있는 과대적합(Overfitting)을 방지
# model.add(Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)))   # L2로 과적합방지. 모델에 규제 부여
# model.add(BatchNormalization())        
# 배치 정규화는 평균과 분산을 조정하는 과정이 별도의 과정으로 떼어진 것이 아니라, 신경망 안에 포함되어 학습 시 평균과 분산을 조정하는 과정 역시 같이 조절된다
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val,y_val), verbose=2)

# 시각화
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

model.save('tf22model.hdf5')

del model

mymodel = tf.keras.models.load_model('tf22model.hdf5')

print(x_test[:1], x_test[:1].shape)
plt.imshow(x_test[:1].reshape(28,28), cmap="Greys")
plt.show()

# 이미지 분류 예측
pred = mymodel.predict(x_test[:1])
print('pred : ', pred)

print('예측값 : ', np.argmax(pred, 1))

print('실제값 : ', y_test[:1])
print('실제값 : ', np.argmax(y_test[:1], 1))




