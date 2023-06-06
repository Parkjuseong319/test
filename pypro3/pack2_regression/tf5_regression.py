# cost를 최소화 하기
# y(예측값) = w * x + b

# 예측값과 실제값의 차이에 따라 cost 크기가 다름
import math
import numpy as np
"""
real = [10, 9, 3, 2, 11]
# pred = [11, 5, 2, 4, 3]     # 모델에 의해 예측된 결과
pred = [11, 5, 4, 3, 11]

cost = 0

for i in range(5):
    cost += math.pow(pred[i] - real[i], 2)         # (예측값 - 실제값 ) 제곱의합 / 전체 수
    print(cost)
    
print(cost / len(pred))
"""
import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3,4,5]     # feature
# y = [1,2,3,4,5]     # label
y = [2,4,6,8,10]     # label
b = 0

# 시각화를 위해 변수 준비
w_val = []
cost_val = []

for i in range(-30, 50):
    feed_w = i * 0.1
    hypothesis = tf.multiply(feed_w, x)  + b           # y = wx + b
    cost =  tf.reduce_mean(tf.square(hypothesis - y))            # (예측값 - 실제값 ) 제곱의합 / 전체 수
    # cost = tf.reduce_sum(tf.pow(hypothesis - y, 2)) / len(y)
    cost_val.append(cost)
    w_val.append(feed_w)
    print(str(i) + ' ' + 'cost : ' + str(cost.numpy()) + ', weight : ' + str(feed_w))
    
# plt.plot(w_val, cost_val, 'o')
# plt.xlabel('weight')
# plt.ylabel('cost')
# plt.show()

print('선형회귀 모델 생성. keras 모듈 사용 안함')
tf.random.set_seed(2)
w = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))
print(w.numpy(), ' ', b.numpy())

opti = tf.keras.optimizers.SGD()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w, x), b)
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))
    grad = tape.gradient(loss, [w,b])   # 자동 미분 (loss 를 w와 b로 미분)
    opti.apply_gradients(zip(grad, [w,b]))
    return loss

x = [1.,2.,3.,4.,5.]
y = [1.2, 2.0, 3.0, 3.5, 5.5]
print(np.corrcoef(x, y))        # 0.9749

w_vals = []
cost_vals = []

for i in range(1, 101):
    loss_val = train_step(x, y)
    cost_vals.append(loss_val.numpy())
    w_vals.append(w.numpy())
    if i % 10 == 0: print(loss_val)

print(cost_vals)
print(w_vals)

plt.plot(w_vals, cost_vals, 'o')
plt.xlabel('weight')
plt.ylabel('cost')
plt.show()

print('cost 최소일때 w : ', w.numpy())
print('cost 최소일때 b : ',b.numpy())
# y = 0.9893332 * x + 0.08461335

# 예측
new_x = [1, 2, 3.5, 9.0]
new_pred = tf.multiply(new_x, w) + b
print('예측 결과 : ', new_pred.numpy())

y_pred = tf.multiply(x, w) + b
plt.plot(x, y, 'ro', label = 'real')
plt.plot(x, y_pred, 'b-', label = 'pred')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
