# 패션 MNIST로 이미지 분류 모델 작성
# 지금까지 여러가지를 커스텀 했는데 이것들을 한 이유는 파이썬으로 OOP하는 방법을 알려주기 위함이다.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from pack3_classification import tf23_earlystop

(x_train, y_train), (x_test, y_test) =tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(x_train[0])   # feature
# print(y_train[0])   # label
class_names = ['T-shirt/top',  'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', "Shirt", 'Sneaker', 'Bag', 'Ankle boot']

print(set(y_train))     # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(class_names[y_train[i]])
#     plt.imshow(x_train[i])
# plt.show()
    
x_train = x_train / 255.0
x_test = x_test / 255.0
# print(x_train[0])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(patience=3, mode='auto', monitor='loss')

my_callback = tf23_earlystop.MyEarlyStopping()      # 별도 작성한 Callback sub class 호출
model.fit(x_train, y_train, batch_size = 128, epochs=10, verbose=2, callbacks=[my_callback])

test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_loss : ', test_loss)
print('test_acc : ', test_acc)

pred = model.predict(x_test)

print("예측값 : ", np.argmax(pred[0]))
print("실제값 : ", y_test[0])

# 이미지 출력용 함수
def plot_image(i, pred_arr, true_label, img):
    pred_arr, true_label, img = pred_arr[i], true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')
    
    pred_label = np.argmax(pred_arr)
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[pred_label], 100 * max(pred_arr), class_names[true_label]), color=color)
    
# i=10
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, pred, y_test, x_test)
# plt.show()
 

# 이미지 분류 정도 정보 및 막대 그래프 출력용 함수
def plot_valu_bar(i, pred_arr, true_label):
    pred_arr, true_label = pred_arr[i], true_label[i] 
    chart = plt.bar(range(10), pred_arr)
    plt.ylim([0,1])
    pred_label = np.argmax(pred_arr)
    chart[pred_label].set_color('red') 
    chart[true_label].set_color('blue') 

# plt.subplot(1,2,2)
# plot_valu_bar(i, pred, y_test)
# plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

plt.figure(figsize=(12,10))
for i in range(num_images):
    plt.subplot(num_rows, num_cols * 2, i * 2+1)
    plot_image(i, pred, y_test, x_test)
    plt.subplot(num_rows, num_cols * 2, i * 2+2)
    plot_valu_bar(i, pred, y_test)
    plt.yticks([])
    
plt.show()

# 하나의 이미지에 대해 전체 레이블 중 확률 값을 시각화
img = x_test[0]
print(img.shape)    # (28, 28)
img = (np.expand_dims(img, 0))
print(img.shape)    # (1, 28, 28)

pred_single = model.predict(img)
print('pred_single', pred_single)
plot_valu_bar(0, pred_single, y_test)
plt.xticks(range(10), class_names, rotation = 45)
plt.show()

print(np.argmax(pred_single[0]))
