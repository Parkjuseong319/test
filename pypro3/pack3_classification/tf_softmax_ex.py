# 문제3) BMI 식으로 작성한 bmi.csv 파일을 이용하여 분류모델 작성 후 분류 작업을 진행한다.
# https://github.com/pykwon/python/blob/master/testdata_utf8/bmi.csv
# train/test 분리 작업을 수행.
# 평가 및 정확도 확인이 끝나면 모델을 저장하여, 저장된 모델로 새로운 데이터에 대한 분류작업을 실시한다.
# EarlyStopping, ModelCheckpoint 사용.
# 새로운 데이터, 즉 키와 몸무게는 키보드를 통해 입력하기로 한다. fat, normal, thin으로 분류

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

data = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/bmi.csv',skiprows=1, delimiter=',', dtype=object)
print(data[:3])

x_data = data[:, 0:2]
x_data = x_data.astype(float)

label = LabelEncoder()
label_fit = label.fit_transform(data[:, -1])
print(label_fit)
y_data = tf.keras.utils.to_categorical(label_fit, num_classes=3)
print(x_data[:2])
print(y_data[:2])

x_train, x_test,y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape,y_train.shape, y_test.shape )


# model 생성
early_stop = EarlyStopping(monitor='val_loss', mode='auto', patience=4)
modelchk = ModelCheckpoint(filepath='./bmimodel', monitor='val_loss', verbose=0, save_best_only= True)

model = Sequential()
model.add(Dense(units=16, input_shape=(2,), activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[early_stop, modelchk])


# test
pred = model.predict(x_test)
print('예측 - ', pred[:3])
print('실제 - ', y_test[:3].flatten())


# 새로운 값 예측
height = float(input('키를 입력하세요'))
weight = float(input('몸무게를 입력하세요'))
new_data = np.array([[height, weight]])

new_pred = model.predict(new_data)
re = np.argmax(new_pred.flatten())
print("당신의 몸 상태는 {}입니다.".format('fat' if re == 0 else 'normal' if re == 1 else 'thin'))

