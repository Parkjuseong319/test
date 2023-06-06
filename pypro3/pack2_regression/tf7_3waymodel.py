# 3 ways to create a Keras model with TensorFlow 2.0
# (Sequential, Functional, and Model  Subclassing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

# 공부 시간에 따른 성적 결과 예측
x_data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y_data = np.array([11, 32, 59, 66, 70], dtype=np.float32)
print(np.corrcoef(x_data.flatten(), y_data))        # 0.95126

# 단순 성형회귀 모델 작성 방법1
print('Sequential api 사용 : 가장 단순한 구조. ')
model = Sequential()
model.add(Dense(units=2, input_dim=1, activation='linear'))
model.add(Dense(units=1, activation='linear'))
print(model.summary())

opti = optimizers.Adam(learning_rate= 0.1)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])
hitory = model.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model.evaluate(x=x_data, y=y_data)
print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print("설명력 : ", r2_score(y_data, model.predict(x_data)))     # 0.9045
print('실제값 : ', y_data)
print('예측값 : ', model.predict(x_data).flatten())    # r2_score는 온전히 신뢰 불가. accuracy가 더 신뢰도가 높다.

# new_data = [[1.5], [2.3], [5.8]]
new_data = [1.5, 2.3, 5.8]
new_data = np.expand_dims(new_data, axis=1)     # 차원 증가
print(new_data)
print(new_data[:, np.newaxis])      # 슬라이싱 할 때 np.newaxis를 하면 차원 증가.
print('새로운 예측값 : ', model.predict(new_data).flatten())  # ravel()  차원 떨어트리기

"""
# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

plt.plot(x_data.ravel(), model.predict(x_data), 'b', x_data.ravel(), y_data, 'ko')
plt.xlabel('공부시간')
plt.ylabel("점수")
plt.show()

# MSE의 변화 추이를 시각화
plt.plot(hitory.history['mse'], label='평균제곱오차')
plt.xlabel('학습횟수')
plt.show()
"""

print('----------' * 10)
# 단순 성형회귀 모델 작성방법2
print('functional api 사용 : 유연한 구조. 입력데이터로부터 여러층 공유, 다양한 종류의 입출력')
from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(1,))      # Input class로 입력층 생성. 여러개 생성 가능하다.
output1 = Dense(units=2, activation='linear')(inputs)       # 이전 레이어를 다음 레이어에 할당한다. 은닉층
output2 = Dense(units=1, activation='linear')(output1)      # 출력층
model2 = Model(inputs, output2)
print(model2.summary())

model2.compile(optimizer=opti, loss='mse', metrics=['mse'])
hitory = model2.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model2.evaluate(x=x_data, y=y_data)
print('loss_metrics2 : ', loss_metrics)

print("설명력 : ", r2_score(y_data, model2.predict(x_data)))     # 0.9045
print('실제값 : ', y_data)
print('예측값 : ', model2.predict(x_data).flatten())

print("---------" * 10)
# 단순 선형회귀 모델 작성방법3
print('Model Subclassing 사용 : 동적인 구조. 고난이도의 작업에서 활용성이 높다.')

class MyModel(Model):
    def __init__(self):     # 생성자에서 레이어 생성함
        super(MyModel, self).__init__()
        self.d1 = Dense(units=2, activation='linear')
        self.d2 = Dense(units=1, activation='linear')
        
    def call(self, x):     # 모델에 call이란 함수가 내부에 존재하는데 fit, compile 할때 callback은 자동으로 호출된다.
        inputs = self.d1(x)     # input layer
        return self.d2(inputs)  # 이전 레이어 다음 레이어에 할당해줌
    
model3 = MyModel()

model3.compile(optimizer=opti, loss='mse', metrics=['mse'])
hitory = model3.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model3.evaluate(x=x_data, y=y_data)
print('loss_metrics2 : ', loss_metrics)

print("설명력 : ", r2_score(y_data, model3.predict(x_data)))     # 0.9045
print('실제값 : ', y_data)
print('예측값 : ', model3.predict(x_data).flatten())

print('\nModel Subclassing 사용2 : 동적인 구조. 고난이도의 작업에서 활용성이 높다.')
from keras.layers import Layer
import tensorflow as tf
class Linear(Layer):
    def __init__(self, units=1):        # 사용자 정의층 : 새로운 연산을 위한 레이어 혹은 편의를 위해 여러 레이어를 하나로 묶은 레이어 구현할 때 사용한다.
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):   # 모델 가중치 관련 내용을 기술
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)  # trainable=True는 역전파를 사용하라는 의미
        self.b = self.add_weight(shape=(self.units, ), initializer='zeros', trainable=True)
    
    def call(self, inputs):
        # 정의된 값으로 해당층의 로직을 정의
        return tf.matmul(inputs, self.w) + self.b
    
class MyLinearModel(Model):
    def __init__(self):     # 생성자에서 레이어 생성함
        super(MyLinearModel, self).__init__()
        self.linear1 = Linear(2)
        self.linear2 = Linear(1)
    
    def call(self, inputs):
        x = self.linear1(inputs)
        return self.linear2(x)
    
model4 = MyLinearModel()

model4.compile(optimizer=opti, loss='mse', metrics=['mse'])
hitory = model4.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model4.evaluate(x=x_data, y=y_data)
print('loss_metrics4 : ', loss_metrics)

print("설명력4 : ", r2_score(y_data, model4.predict(x_data)))     # 0.9039
print('실제값4 : ', y_data)
print('예측값4 : ', model4.predict(x_data).flatten())
