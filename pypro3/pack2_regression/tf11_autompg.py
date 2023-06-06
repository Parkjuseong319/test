# 다중선형 회귀 - 자동차 연비 예측

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
import tensorflow as tf

from keras import layers

dataset = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/auto-mpg.csv')
print(dataset.head(2))
print(dataset.columns)
del dataset['car name']
print(dataset.info())
print(dataset.corr())
dataset.drop(['cylinders', 'acceleration', 'model year', 'origin'], axis='columns', inplace=True)
print(dataset.info())
print(dataset.isna().sum())

dataset = dataset.apply(lambda x:x.replace('?', 0))
# dataset = dataset.drop(dataset[:,[dataset['horsepower']== '']] , axis=0)
dataset['horsepower'] = dataset['horsepower'].apply(lambda x:float(x))
dataset = dataset.dropna()
print(dataset.info())

# sns.pairplot(dataset[['mpg', 'displacement', 'horsepower', 'weight']], diag_kind='kde')
# plt.show()

# train / test split 
train_dataset = dataset.sample(frac=0.7, random_state=123)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset[:2], train_dataset.shape)
print(test_dataset[:2], test_dataset.shape)

# 표준화 작업 준비 ------------------------------------------
train_stat = train_dataset.describe()
train_stat.pop('mpg')
# print(train_stat)
train_stat = train_stat.transpose()
print(train_stat)


def std_func(x):        # 표준화 처리함수 (요소값 - 평균) / 표준편차
    return (x -train_stat['mean']) / train_stat['std']
# ------------------------------------------------------

# print(std_func(train_dataset[:3))
st_train_data = std_func(train_dataset)
st_train_data = st_train_data.drop(['mpg'], axis='columns')
print(st_train_data[:2])
st_test_data = std_func(test_dataset)
st_test_data = st_test_data.drop(['mpg'], axis='columns')
print(st_test_data[:2])

train_label = train_dataset.pop('mpg')
print(train_label[:2])
test_label = test_dataset.pop('mpg')
print(test_label[:2])

print()
from keras.models import Sequential
from keras.layers import Dense

def build_model():
    network = Sequential([
        Dense(units=32, activation='relu', input_shape=[3]),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='linear')
    ])
    
    opti = tf.keras.optimizers.Adam(0.01)
    network.compile(optimizer=opti, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
    # MSE는 제곱의 합을 평균냈다라고 하면, MAE는 오차의 절대값의 합을 평균낸 지표이다.
    # 제곱을 한 MSE와는 달리 절대값을 평균낸 값이기 때문에 전체 흐름을 쉽게 파악할 수 있다. 
    return network

model = build_model()
print(model.summary())

epochs = 1000
# 조기종료
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(st_train_data, train_label, batch_size=32, epochs=epochs, validation_split=0.2, verbose=2, callbacks=[early_stop])

df = pd.DataFrame(history.history)
print(df.head(3))
print(df.columns)

# 모델 학습 정보 시각화
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(8,12))
    
    plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('mean absolute error [mpg]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='train err')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='validation err')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('mean squared error [$mpg^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='train err')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='validation err')
    plt.legend()
    plt.show()
    
plot_history(history)
    
# 모델 평가
loss, mse, mae = model.evaluate(st_test_data, test_label)
print('test dataset으로 평가 mae : {:5.3f}'.format(mae))
print('test dataset으로 평가 mse : {:5.3f}'.format(mse))
print('test dataset으로 평가 loss : {:5.3f}'.format(loss))

from sklearn.metrics import r2_score
print('결정계수 : ', r2_score(test_label, model.predict(st_test_data)))

print()
# 새로운 값으로 예측
new_data = pd.DataFrame({'displacement':[300, 400], 'horsepower':[120, 140], 'weight':[2000, 4000]})

# 표준화 데이터로 학습했으므로 new_data도 표준화 작업을 해야한다.
new_st_test_data = std_func(new_data)
new_test_pred = model.predict(new_st_test_data).flatten()
print('예측결과 : ', new_test_pred)






