# 로이터 뉴스 분류하기(Reuters News Classification)
# 케라스에서 제공하는 로이터 뉴스 데이터를 LSTM을 이용하여 텍스트 분류를 진행해보겠습니다. 
# 로이터 뉴스 기사 데이터는 총 11,258개의 뉴스 기사가 46개의 뉴스 카테고리로 분류되는 뉴스 기사 데이터입니다.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters

(train_feature, train_label), (test_feature, test_label) = reuters.load_data(num_words=10000, test_split=0.2)
# 자주 나타나는 단어 10,000개를 사용한다는것.

print(train_feature.shape, train_label.shape)   # (8982,) (8982,)
print(test_feature.shape, test_label.shape)     # (2246,) (2246,)
print(train_feature[:1])    
print(train_label[:1], set(train_label))   
"""   
# 참고로 실제 뉴스 데이터 보기
word_index = reuters.get_word_index()
print(word_index)
reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])
print(train_feature[0])
decord_review = ' '.join([reverse_word_index.get(i) for i in train_feature[0]])
print(decord_review)
# ----------------------------------------------------------------
"""

# 데이터 준비
def vector_seq(sequences, dim=10000):       # list -> vector
    results = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results

x_train = vector_seq(train_feature)
print(x_train)
print(x_train.shape)
x_test = vector_seq(test_feature)

# label
"""
def to_one_hot(labels, dim = 46):       # keras의 to_categorical을 사용해도 된다. 46은 label의 종류개수.
    results = np.zeros((len(labels), dim))
    for i, seq in enumerate(labels):
        results[i, seq] = 1.
    return results
    
one_hot_train_labels = to_one_hot(train_label)
one_hot_test_labels = to_one_hot(test_label)
print(one_hot_train_labels[0])
"""

from keras.utils import to_categorical
one_hot_train_labels = to_categorical(train_label)
one_hot_test_labels = to_categorical(test_label)
print(one_hot_train_labels[0])

# model
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(x_train, one_hot_train_labels, epochs=10, batch_size=128, validation_split=0.2, verbose=2)

# 훈련검정 용 dataset을 준비 : 1000개의 샘플을 떼어 validation set을 사용
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
print(len(x_val), len(partial_x_train))

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
print(len(y_val), len(partial_y_train))

history = model.fit(partial_x_train, partial_y_train, epochs=200, batch_size=128, validation_data=(x_val, y_val), verbose=2)

# 시각화 : 훈련과 검증 정보 확인
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='train acc')
plt.plot(epochs, val_acc, 'r', label='validation acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

print('evaluate : ', model.evaluate(x_test, one_hot_test_labels))

# pred
pred = model.predict(x_test)
print(pred[0].shape)
print(np.sum(pred[0]))
print('예측값 : ', np.argmax(pred[0]))
print('실제값 : ', test_label[0])
