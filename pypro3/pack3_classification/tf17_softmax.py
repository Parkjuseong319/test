# 다-항분류는 softmax 함수를 사용

# 참고
"""
import numpy as np
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c) # 오버플로 방지
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    
    return y

imsi = np.array([2, 1,0])
print(softmax(imsi))
print(np.argmax(softmax(imsi)))
"""
# 다항분류 : network의 최종 활성화함수(softmax)로 인해 결과가 확률값으로 여러 개가 출력.
# 이 때 확률값이 가장 큰 인덱스를 분류의 결과로 취함
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical  # OneHotEncode와 같은 역할

np.random.seed(1)

# dataset을 준비 (시험결과라고 가정)
xdata = np.random.random((1000, 12))
ydata = np.random.randint(5, size=(1000, 1))    # 범주 5가지. 국어:0 ~ 체육:0 라고 가정
print(xdata[:2])
print(ydata[:2])
ydata = to_categorical(ydata, num_classes=5)
print(ydata[:2])
print([np.argmax(i) for i in ydata[:2]])

# 모델
model = Sequential()
model.add(Dense(units=32, input_shape=(12,), activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
print(model.summary())

# 'categorical_crossentropy' : 다항 분류 일때 사용한다. 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(xdata, ydata, epochs=1000, batch_size = 32, verbose=2)
model_eve = model.evaluate(xdata, ydata, batch_size=32)
print('model_evaluate : ', model_eve)

# 시각화 
plt.plot(history.history['loss'], label='loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

# 분류 예측 결과
print('예측값 : ', model.predict(xdata[:5]))
print('예측값 : ', [np.argmax(i) for i in model.predict(xdata[:5])])
print('실제값 : ', ydata[:5])
print('실제값 : ', [np.argmax(i) for i in ydata[:5]])
classes = np.array(['국어', '영어', '수학', '과학', '체육'])
print('예측값 : ', classes[np.argmax(model.predict(xdata[:5]), axis=-1)])
print('실제값 : ', classes[np.argmax(ydata[:5], axis=-1)])