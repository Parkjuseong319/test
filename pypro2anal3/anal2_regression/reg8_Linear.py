# 선형회귀모델 평가 지표(score) 알아보기

from sklearn.linear_model import LinearRegression   # summary() 지원 X
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 공부 시간에 따른 시험점수 자료 사용
df = pd.DataFrame({'studytime':[3,4,5,8,10,5,8,6,3,6,10,9,7,0,1,2], \
                   'score':[76,74,74,89,92,75,84,82,73,81,89,88,83,40,70,68]})
print(df)

# 학습된 모델의 성능을 확인하기 위해 dataset을 분리 작업(train(학습데이터)/test(검정데이터))
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.4, random_state=12)  # 6:4로 나눈다는 것. random_state=12는 난수표 얘기하는것
print(df.shape)     # (16, 2)
print(train.shape, test.shape)  # (9, 2) (7, 2)

x_train = train[['studytime']]
y_train = train['score']
x_test = test[['studytime']]
y_test = test['score']
print(x_train)
print(y_train)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (9, 1) (7, 1) (9, 1) (7, 1)

# 모델 생성
model = LinearRegression().fit(x_train, y_train)    # 모델학습은 train 사용
y_pred = model.predict(x_test)      # 모델 검정은 test 사용
print('실제값 : ',y_test.values)
print('예측값 : ',y_pred)

# 모델 성능을 수치로 표현
print('결정계수 : ', r2_score(y_test, y_pred))  # 0.32995875214  << 실무에선 쓸만한 정도이다.

