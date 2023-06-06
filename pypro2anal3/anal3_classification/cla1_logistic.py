# Logistic Regression(분류모델)    < 이항분류
# 독립변수 : 연속형, 종속변수 : 범주형
# 모델의 출력값을 logit변환(log10(odds ratio)) 후 sigmoiid function을 사용하여 0 ~ 1 사이의 실수값 생성
# 0.5를 기준으로 이항분류

import math

# sigmoid function 실습
def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))

print(sigmoidFunc(3))       # 인자값은 logit 변환된 수치.
print(sigmoidFunc(-3.789))
print(sigmoidFunc(-2))
print(sigmoidFunc(7))
print(sigmoidFunc(1))
print(sigmoidFunc(-0.1))

print("Logistic Regression을 이용한 분류모델 작성")
# mtcars dataset 사용
import statsmodels.api as sm

mtcardata = sm.datasets.get_rdataset('mtcars')
print(mtcardata.keys())
mtcars = sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(3), mtcars.shape)     # (32, 11)

mtcar = mtcars.loc[:, ['mpg', 'hp', 'am']]      # am은 수동 or 자동을 보여줌. 범주형 데이터
print(mtcar.head(3))
print(mtcar['am'].unique())     # [1 0] 이항분류 되어있음.


# 연비와 마력수에 따른 변속기 분류(수동, 자동)
# 모델 생성 방법1
import statsmodels.formula.api as smf

formula = "am ~ hp+mpg"
result = smf.logit(formula=formula, data = mtcar).fit()
print(result.summary())     # Logit Regression Results

import numpy as np
pred = result.predict(mtcar[:10])
print('예측값 : ', pred.values)
print('예측값 : ', np.around(pred.values))     # np.around() : 0.5 기준으로 0,1을 출력

print('실제값 : ', mtcar['am'][:10].values)

print()
conf_tab = result.pred_table()
print("confusion matrix : \n",conf_tab)
print('분류 정확도 : ', (16 + 10)/ len(mtcar))   # 0.8125. accuracy(분류정확도):81.25%
print('분류 정확도 : ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))  # 0.8125

print()
# 모델 생성 방법2
result2 = smf.glm(formula=formula, data = mtcar, family=sm.families.Binomial()).fit()   # 기본값은 정규분포를 따름. Binomial()은 이항분포.
#glm = generalized linear model
print(result2.summary())
glm_pred = result2.predict(mtcar[:10])
print('예측값 : ', np.around(glm_pred.values))
print('실제값 : ', mtcar['am'][:10].values)

from sklearn.metrics import accuracy_score
glm_pred2 = result2.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glm_pred2.values)))    # 0.8125

print('새로운 hp, mpg에 대한 분류')
import pandas as pd
newdf = pd.DataFrame({'mpg':[10, 30, 50, 5], 'hp':[80, 100, 130, 50]})
new_pred = result2.predict(newdf)
print(new_pred.values)
print(np.around(new_pred.values))       # [0. 1. 1. 0.]
print(np.rint(new_pred.values))     # 소수점 아래 첫번째 자리에서 반올림하여 정수로만 표현하는 함수 rint

# ML의 포용성(inclusion, tolerance) - 과적합(overfitting)이 발생하지 않는 상태에서 최적의 모델을 생성
# 최적화 / 일반화









