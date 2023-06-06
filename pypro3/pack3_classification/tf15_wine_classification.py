import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os 


wdf = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/wine.csv", header=None)
print(wdf.head(3))
print(wdf.info())
print(wdf.iloc[:, 12].unique()) # [1 0]
print(len(wdf[wdf.iloc[:,12] == 0]))    # 4898
print(len(wdf[wdf.iloc[:,12] == 1]))    # 1599
dataset = wdf.values
x = dataset[:, 0:12]    # feature
y = dataset[:, -1]      # label
print(x[:2])
print(y[:2])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=12)

# create model
model = Sequential()
# model.Flatten()     # 입력 데이터의 차원을 낮추는 역할(기본적으로 처리됨)
model.add(Dense(units=32, input_dim=12, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# BinaryCrossentropy()        binary_crossentropy

# fit() 하기 전에 모델 정확도 확인
loss, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('학습하지 않은 모델의 정확도 : {:5.2f}'.format(acc * 100))    # 76.62

print()
# 학습 도중 모델을 저장 가능
MODEL_DIR = './mymodel/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# modelPath = '{epoch:02d}.h5'        # 파일명을 이렇게 지정해주면 여러개가 생성된다. 불필요한게 생성됨
# chkPoint = ModelCheckpoint(filepath= MODEL_DIR + modelPath, monitor='val_loss', verbose=0)
modelPath = 'tf15best.h5'           # 하나의 이름을 지정해주면 하나의 파일에서 갱신되면서 저장된다.
chkPoint = ModelCheckpoint(filepath= MODEL_DIR + modelPath, monitor='val_loss', verbose=0, save_best_only= True)
# save_best_only= True 는 loss 값이 떨어질때만 저장하게 하는 파라미터이다.
# 가장 성능이 우수한 모델만 저장된다.

# 조기종료
early_stop = EarlyStopping(monitor='val_loss', mode='auto', patience=5)

history = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=[early_stop, chkPoint])

# fit() 한 후 모델 정확도 확인
loss, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('학습 후의 모델의 정확도 : {:5.2f}'.format(acc * 100))      # 98.10

# model.save('aaa.h5')

# 시각화
epoch_len = np.arange(len(history.epoch))

plt.plot(epoch_len, history.history['val_loss'], c='red', label='val_loss')
plt.plot(epoch_len, history.history['loss'], c='blue', label="loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

plt.plot(epoch_len, history.history['val_accuracy'], c='red', label='val_accuracy')
plt.plot(epoch_len, history.history['accuracy'], c='blue', label="accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

# 저장된 모델 읽기
from keras.models import load_model
ourmodel = load_model(MODEL_DIR + 'tf15best.h5')
new_data = x_test[:5, :]       # 새로운 자료로 해야하지만 기존 자료로 분류 예측
print(new_data)
pred = ourmodel.predict(new_data)
np.set_printoptions(suppress=True)
# print('결과 : ', pred.astype(int))
print('결과 : ', np.where(pred > 0.5, 1, 0).ravel())






