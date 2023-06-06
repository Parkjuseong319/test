# 선형 회귀분석 : 각각의 데이터에 대한 잔차제곱합이 최소가 되는 추세선을 만들고, 이를 통해
# 독립변수(x:연속형)가 종속변수(y:연속형)에 얼마나 영향을 주는지 인과관계를 분석. 
# 기본 충족 조건 : 선형성, 잔차정규성, 잔차독립성, 등분산성(다중회귀), 다중공선성(다중회귀)
# 과거의 데이터를 기계학습 한 후 정량적인 모델을 생성

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

# 모델 생성 맛보기
# 방법1 : make_regression을 사용. model 생성 x
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x[:5])
print(y[:5])
print(coef)
# plt.scatter(x, y)
# plt.show()
# 회귀식 y = wx + b    y = 89.47430739278907 * x + 100 
# 새로운 값 예측 가능.
y_pred = 89.47430739278907 * -1.70073563 + 100
print('y_pred : ',y_pred)
y_pred = 89.47430739278907 * -0.67794537 + 100
print('y_pred : ',y_pred)
# 미지의 새로운 값 예측
y_pred_new = 89.47430739278907 * 33 + 100
print('y_pred_new : ',y_pred_new)

xx =x
yy = y

print()
# 방법2 : Linear Regression을 사용. model 생성 O
# 머신러닝은 귀납적 추론방식을 따른다.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# print(model)
fit_model = model.fit(xx, yy)       # 학습 데이터로 모델(모형) 추정 : 절편과 기울기를 얻게된다.
print('기울기 :',fit_model.coef_)     # [89.47430739]
print('절편 :',fit_model.intercept_) # 100.0
# 회귀식 y = wx + b    y = 89.47430739 * x + 100 
# 예측값 확인 함수 사용
y_new = fit_model.predict(xx[[0]])  # 기존자료로 모델 검정
print('y_new : ', y_new)
y_new = fit_model.predict([[5]])  # 새로운 자료로 예측. 2차원으로 학습 시켰으면 2차원으로 값을 넣어줘야한다. 차원에 맞게 값 넣어주는거 중요.
print('y_new : ', y_new)

print()
# 방법3 : ols를 사용. model 생성 O    ols는 1차원 값만.
import statsmodels.formula.api as smf
import pandas as pd
print(np.shape(xx))     # (50, 1)
x1 = xx.flatten()       # 차원 축소시키는 함수. python analysis 51번 게시글 참조
print(x1.shape)
y1 = yy
print(y1.shape)

data = np.array([x1,y1])
df = pd.DataFrame(data.T, columns=['X', 'Y'])
print(df.head(3))

model2 = smf.ols(formula = 'y1 ~ x1', data = df).fit()
print(model2.summary())     # OLS Regression Results를 제공

# 예측 
print(x1[:2])
new_df = pd.DataFrame({'x1':[-1.700736, -0.677945]})
new_pred = model2.predict(new_df)
print(new_pred)

# 새로운 값으로 예측
new2_df = pd.DataFrame({'x1':[-1.000, 12]})
new2_pred = model2.predict(new2_df)
print(new2_pred)

# 모델 생성 후 fit()으로 학습하고 predict()으로 예측 결과를 얻음.
