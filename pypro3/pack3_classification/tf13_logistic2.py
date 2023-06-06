import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

np.random.seed(0)

x = np.array([[1,2],[2,3],[3,4],[4,3],[3,2],[2,1]])
y = np.array([[0],[0],[0],[1],[1],[1]])

model = Sequential([
    Dense(units = 1, input_dim=2),  # input_shape=(2,)
    Activation('sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=1000, batch_size=1, verbose=1)
meval = model.evaluate(x, y)
print(meval)  # [0.209698(loss), 1.0(정확도)]

pred = model.predict(np.array([[1,2],[10,5]]))
print('예측 결과 : ', pred)     # [[0.16490099] [0.9996613 ]]
print('예측 결과 : ', np.squeeze(np.where(pred > 0.5, 1, 0)))  # [0 1]