# 다항회귀(Polynomial Regression) - 비선형회귀 : 데이터의 경향이 선형으로 표현할 수 없을때 사용
# 회귀선이 2차, 3차 함수 등의 곡선이 됨
# 다항회귀는 독립변수에 대한 차수를 확장해가며 단항식이 아닌 2차, 3차 등의 회귀모델을 도출한다.

import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
print(len(population_inc))
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# plt.plot(population_inc,population_old,'bo')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

# 지역별 인구증가율과 고령인구비율 : 이상(극단)치 제거 - 세종시 데이터
population_inc = population_inc[:5] + population_inc[6:]  # 5번째는 제외
population_old = population_old[:5] + population_old[6:]
print(len(population_inc))

# plt.plot(population_inc,population_old,'bo')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

print('최소제곱법으로 회귀선 구하기')
# population_inc(x), population_old(y)의 평균
x_bar = sum(population_inc) / len(population_inc)
y_bar = sum(population_old) / len(population_old)

a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(population_old, population_inc))])
a /= sum([(x - x_bar) ** 2 for x in population_inc])
b = y_bar - a * x_bar
print('a: ', a, 'b : ', b)  # y_hat = -0.355834147915461 * x + 15.669317743971302

import numpy as np
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x + b

plt.plot(line_x, line_y, 'r-')
plt.plot(population_inc,population_old,'bo')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()

print('\n최소제곱법을 사용하지 않고 tf를 사용해 회귀선 구하기 : cost를 최소화하기 위해 경사하강법을 사용')

import tensorflow as tf
import random
a = tf.Variable(random.random())     
b = tf.Variable(random.random())     

# 잔차 제곱의 평균 구하는 함수
def comput_loss():
    y_pred = a * population_inc + b     # y_hat = ax + b
    loss = tf.reduce_mean((population_old - y_pred)**2)
    return loss

opti = tf.keras.optimizers.Adam(learning_rate=0.5)
for i in range(1, 1001):
    opti.minimize(comput_loss, var_list=[a,b])      # 잔차제곱의 평균을 최소화
    if i % 100 == 0:
        print('a:', a.numpy(), ', b:', b.numpy(), ', loss:', comput_loss().numpy())

        
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x + b

plt.plot(line_x, line_y, 'r-')
plt.plot(population_inc,population_old,'bo')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()
    
    
print("\n다항회귀 : 다항식 사용 - tensorflow를 이용하여 2차함수 회귀선 구하기")
# ax² + bx + c
a = tf.Variable(random.random())     
b = tf.Variable(random.random())     
c = tf.Variable(random.random())     

# 잔차 제곱의 평균 구하는 함수
def comput_loss2():
    y_pred = a * population_inc * population_inc + b * population_inc + c    # y_hat = ax² + bx + c
    loss = tf.reduce_mean((population_old - y_pred)**2)
    return loss

opti = tf.keras.optimizers.Adam(learning_rate=0.05)
for i in range(1, 1001):
    opti.minimize(comput_loss2, var_list=[a,b,c])      # 잔차제곱의 평균을 최소화
    if i % 100 == 0:
        print('a:', a.numpy(), ', b:', b.numpy(), ', c:', c.numpy(),  ', loss:', comput_loss().numpy())

        
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x * line_x + b * line_x + c

plt.plot(line_x, line_y, 'r-')
plt.plot(population_inc,population_old,'bo')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()

print('\n다항 회귀 : Sequential을 이용하여 회귀 모델 작성')
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)      # linear 생략 가능
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01), loss='mse')
print(model.summary())

model.fit(population_inc, population_old, epochs=1000)
print(model.predict(population_inc).flatten())

line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = model.predict(line_x).flatten()

plt.plot(line_x, line_y, 'r-')
plt.plot(population_inc,population_old,'bo')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()





