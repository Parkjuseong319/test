# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

student = pd.read_csv('../testdata/student.csv')
# print(student.head(5))
# print(student.corr())        # 강력한 상관관계에 있음을 확인 가능

#   - 국어 점수를 입력하면 수학 점수 예측
model1 = smf.ols('수학~국어',data=student).fit()
# print(model1.summary())
# y = 0.5705 * 국어 + 32.1069
# print(0.5705 * 90 + 32.1069)    # 83.4519
# print(model1.predict(pd.DataFrame({'국어':[85]})))    # 83.456401
# kor = int(input('국어점수를 입력하시오'))
# print("당신의 대략의 수학점수 : ", np.round(model1.predict(pd.DataFrame({'국어':[kor]})).values,2))

#   - 국어, 영어 점수를 입력하면 수학 점수 예측

model2 = smf.ols('수학~국어+영어',data=student).fit()
# print(model2.summary())
# y = 0.1158 * 국어 + 0.5942 * 영어 + 22.6238
# print(0.1158 * 90 + 0.5942 * 85 + 22.6238)    # 83.5528
# print(model2.predict(pd.DataFrame({'국어':[90], '영어':85})))    # 83.55347
# kor = int(input('국어점수를 입력하시오'))
# eng = int(input('영어점수를 입력하시오'))
# print("당신의 대략의 수학점수 : ", np.round(model2.predict(pd.DataFrame({'국어':[kor], '영어':eng})).values[0],2))


from sklearn.linear_model import LinearRegression


# 다중선형회귀 모델 생성 및 피팅
x = student[['국어', '영어']]
y = student['수학']
model = LinearRegression()
model.fit(x, y)

# 회귀평면을 그리기 위한 격자 생성
x1_range = np.linspace(x['국어'].min(), x['영어'].max(), 100)
x2_range = np.linspace(x['국어'].min(), x['영어'].max(), 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# 격자에 대한 예측값 계산
x_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
y_grid = model.predict(x_grid).reshape(x1_grid.shape)

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 산점도 그리기
ax.scatter(student['국어'], student['영어'], student['수학'], color='blue', label='Actual Data')

# 회귀평면 그리기
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color='red', label='Regression Plane')

# 축 레이블 설정
ax.set_xlabel('국어')
ax.set_ylabel('영어')
ax.set_zlabel('수학')

plt.show()

