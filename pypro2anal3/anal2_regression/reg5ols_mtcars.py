# 선형회귀분석 : mtcars dataset
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import statsmodels.api

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(mtcars.columns)
# ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear','carb']
print(mtcars.info())
print(mtcars.corr())        # hp, wt, mpg

# 단순 선형회귀 : 독립(hp), 종속(mpg)
# 시각화
plt.scatter([mtcars.hp,mtcars.wt], mtcars.mpg)
plt.xlabel("마력수")
plt.ylabel("연비")
plt.show()

model1 = smf.ols('mpg ~ hp', data=mtcars).fit()
print(model1.summary())
# 수식 : y = wx + b        y = -0.0682 * x + 30.0989
print(mtcars.hp[:1])    # 110
print(mtcars.mpg[:1])   # 21.0
print(-0.0682 * 110 + 30.0989)      # 22.5969
print(model1.predict(pd.DataFrame({'hp':[110]})))   # 22.59375
print('새로운 hp 값에 대한 mpg 추정값 구하기')
new_hp = pd.DataFrame({'hp':[88,99,120,150,200]})
print('예측 결과 : ', model1.predict(new_hp).values)    
# 24.09477207, 23.34426101, 21.91146717, 19.86461883, 16.45320493

plt.scatter(mtcars.hp, mtcars.mpg)
plt.plot(mtcars.hp, model1.predict(), color='r')
plt.xlabel("마력수")
plt.ylabel("연비")
plt.show()

print('----'*20)
# 다중선형회귀 : 독립(hp, wt), 종속(mpg)
model2 = smf.ols('mpg ~ hp+wt', data=mtcars).fit()
print(model2.summary())
# 수식 : y = w₁x₁ + w₂x₂ + b        y = -0.0318 * hp + -3.8778 * wt + 37.2273
print(mtcars.hp[:1])    # 110
print(mtcars.wt[:1])    # 2.62
print(mtcars.mpg[:1])   # 21.0
print(-0.0318 * 110 + -3.8778 * 2.62 + 37.2273)      # 23.569464
print(model2.predict(pd.DataFrame({'hp':[110], 'wt':[2.62]})))   #  23.572329
print('새로운 hp, wt 값에 대한 mpg 추정값 구하기')
new_hpwt = pd.DataFrame({'hp':[88,99,120,200,200], 'wt':[1.5,1.8,2.0,1.5,5]})
print('예측 결과 : ', np.round(model2.predict(new_hpwt).values, 1))  
# [28.6 27.1 25.7 25.1 11.5]

