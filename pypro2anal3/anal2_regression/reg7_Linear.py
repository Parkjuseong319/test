# 선형회귀모델 평가 지표(score) 알아보기

from sklearn.linear_model import LinearRegression   # summary() 지원 X
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt

sample_size = 100
np.random.seed(1)

x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:5])
print(y[:5])
scaler = MinMaxScaler()     # 정규화 (0~1사이의 값으로 변환).   독립변수에 대하여 정규화한다.
x_scaled = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled[:5])
print('r : ', np.corrcoef(x, y))
print('r : ', np.corrcoef(x_scaled.flatten(), y))


# plt.scatter(x, y)
# plt.show()

model = LinearRegression().fit(x_scaled, y)     
y_pred = model.predict(x_scaled)
print('예측값 : ', y_pred[:5])
print('실제값 : ', y[:5])

print()
def regScoreFunc(y_true, y_pred):
    print('r2_score(결정계수) :{}'.format(r2_score(y_true, y_pred)))
    print("explained_variance_score(설명분산점수) :{}".format(explained_variance_score(y_true, y_pred)))
    print("mean_squared_error(RMSE-평균제곱근오차) :{}".format(mean_squared_error(y_true, y_pred)))

regScoreFunc(y, y_pred)
# r2_score(결정계수) :0.9987875127274646
# explained_variance_score(설명분산점수) :0.9987875127274646
# mean_squared_error(RMSE-평균제곱근오차) :86.14795101998743
# 결정계수와 설명분산점수가 다르다면 에러에 편향이 있다는 것이고, 이는 모델학습이 잘못됐다는 뜻이다.
# 결정계수 = 설명분산점수. 평균제곱근오차는 작을 수록 좋다.

print('\n표준편차가 크게 다른 x, y 값을 사용')
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30
print(x[:5])
print(y[:5])
scaler = MinMaxScaler()     # 정규화 (0~1사이의 값으로 변환).   독립변수에 대하여 정규화한다.
x_scaled = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled[:5])
print('r : ', np.corrcoef(x, y))
print('r : ', np.corrcoef(x_scaled.flatten(), y))

model2 = LinearRegression().fit(x_scaled, y)     
y_pred2 = model2.predict(x_scaled)
print('예측값 : ', y_pred2[:5])
print('실제값 : ', y[:5])
regScoreFunc(y, y_pred2)
# r2_score(결정계수) :0.2093350679216215
# explained_variance_score(설명분산점수) :0.2093350679216216
# mean_squared_error(RMSE-평균제곱근오차) :282457.9703485092


