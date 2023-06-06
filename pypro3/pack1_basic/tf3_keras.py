# Keras 기본 개념 
  # - 케라스의 가장 핵심적인 데이터 구조는 "모델" 이다.
  # - 케라스에서 제공하는 시퀀스 모델을 이용하여 레이어를 순차적으로 쉽게 쌓을 수 있다. 
  # - 케라스는 Sequential에 Dense 레이어(fully-connected layers 완전히 연결된 레이어)를 쌓는 스택 구조를 사용한다.

import numpy as np  
from keras.models import Sequential     # 버전에 따라 라이브러리 호출 방식이 다르다.
from keras.layers import Dense, Activation 
from keras.optimizers import SGD, RMSprop, Adam


# keras modeling 순서
# 1. 데이터 셋 생성
x = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([[0], [1], [1], [1]])      # or
y = np.array([0, 1, 1, 1])      # 2차원이든 1차원이든 y의 차수는 상관없다.

# 2. 모델 구성
# model = Sequential([
#     Dense(units=1, input_dim=2),        # units는 출력개수, input_dim은 입력개수
#     Activation('sigmoid')        # 활성화함수는 이항분류이므로 sigmoid
# ])

# 다르게 작성하는 방법
model = Sequential()
model.add(Dense(units=1, input_dim=2))
model.add(Activation('sigmoid'))


# 3. 모델 학습 과정 설정, 학습 process 생성
# loss='binary_crossentropy' : 이진 분류일때 사용
# optimizer : cost function의 최소값을 찾는 알고리즘을 말함. 입력데이터와 손실함수를 기반으로 모델을 갱신.
# SGD(Stochastic Gradient Descent, sgd) - 확률적 경사하강법(Cost를 최소화할때 전체가 아닌 일부의 자료만 참여함)
# RMSProp(rmsprop) - SGD 단점(local minimum, 국소최적해)을 보완, 
# Adam(adam) - RMSProp 방향과 스텝 사이즈의 성능을 개선. 가장 성능이 좋다.
# loss : loss function(cost function, 손실함수 ...) - train data로 모델 성능을 특정하는 방법으로 모델이 옳은 방향으로 학습될 수 있도록 한다.
# metrics : 훈련을 모니터링 하고 값을 반환. 정성적 분류 - accuracy, 정량적예측 - mse

# 기본값으로만 optimizer 설정
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# optimizer 커스텀 설정. 클래스를 불러와서 하면 된다.
# model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=RMSprop(learning_rate=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습(더 나은 표현을 찾기를 자동화)시키기
model.fit(x, y, epochs=10, batch_size=1, verbose=0)      # verbose 0(숨기기), 1(자세히), 2(핵심만)
# epochs : 학습 횟수  
# batch_size : 몇개의 문제를 해결 후 학습결과를 확인하는지에 대한 사이즈. 너무 적게주면 학습 결과가 불안정. 보통 4의 배수로 준다. 

# 5. 모델 평가
loss_metrics = model.evaluate(x, y, batch_size=1, verbose=0)
print(loss_metrics) # [loss = 0.5039768218994141, 정확도 = 0.75]

# 6. 학습결과 확인 : 예측값 출력
pred = model.predict(x > 0.5).astype('int32')
print('예측 결과 : ', pred.flatten())

# 7. 모델 저장 및 읽기
# 모델 저장
model.save('./test.hdf5')     # hdf5 확장자는 모델 저장시 주로 사용. 빅데이터 저장할때 사용

# 모델 저장후에는 위의 학습 과정은 더 이상 필요 없다.
# 모델읽기
from keras.models import load_model
mymodel = load_model('test.hdf5')
pred2 = mymodel.predict(x > 0.5).astype('int32')
print('예측 결과 : ', pred2.flatten())




