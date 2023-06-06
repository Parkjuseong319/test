# tensorflow        <= numpy 기반
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.executing_eagerly())

print('상수정의')
print(tf.constant(1))           # 0d tensor : scalar
print(tf.constant([1]))         # 1d tensor 
print(tf.constant([[1]]))       # 2d tensor

print()
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = a + b
print(c, type(c))
c = tf.add(a, b)
print(c)

d = tf.constant([3])
e = c + d       # broadcast 연산
print(e)

print()
print(7)
print(tf.convert_to_tensor(7, dtype=tf.float32))    # 일반 데이터를 tensor로 변환
print(tf.cast(7, dtype=tf.float32))
print(tf.constant(7.0))

import numpy as np
arr = np.array([1, 2])
print(arr, type(arr))
tfarr = tf.add(arr, 5)      # 자동으로 tensor로 형변환
print(tfarr)
print(tfarr.numpy())        # numpy로 형변환
print(np.add(tfarr, 3))


