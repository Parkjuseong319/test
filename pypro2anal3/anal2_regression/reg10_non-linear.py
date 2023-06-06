# 비선형회귀(다항회귀)
# 선형가정이 어긋나는(정규성위반) dataset인 경우 대처방법으로 다항식 항을 추가하여 다항회귀모델 작성
# Non-Linear Regression : 회귀선을 곡선으로 변환해 보다 더 정확하게 데이터 변화를 예측하는데 목적이 있다.

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

# plt.scatter(x,y)
# plt.show()
print(np.corrcoef(x,y))     # 0.48076

# 선형회귀모델
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]        # 차원 증가.

model = LinearRegression().fit(x,y)
ypred = model.predict(x)
print(ypred)    # [2.  2.7 3.4 4.1 4.8]

# plt.scatter(x,y)
# plt.plot(x, ypred, c='r')
# plt.show()            # x가 y를 제대로 설명한다고 보기 어렵다. 
# 해결방법 : 데이터 변환해 모델에 유연성을 주면 된다. 독립변수(feature, 특징, 특성)를 추가해 다항식을 만든다.
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)    
# degree : 열수. 제곱한 값이 새로운 열에 추가된다. 데이터에 따른 적당한 값을 주어야 한다.
x2 = poly.fit_transform(x)      # 특징행렬을 만듦
print(x) 
print(x2) 

model2 = LinearRegression().fit(x2,y)   # 특징행렬값으로 학습.
ypred2 = model2.predict(x2)
print(ypred2)       # [4.14285714 1.62857143 1.25714286 3.02857143 6.94285714]

plt.scatter(x,y)
plt.plot(x, ypred2, c='blue')
plt.show() 

