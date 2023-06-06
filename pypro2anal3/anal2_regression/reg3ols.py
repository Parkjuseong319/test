# 단순 선형회귀 분석(결정론적 선형회귀) : ols() 사용.
import pandas as pd

df = pd.read_csv('../testdata/drinking_water.csv')
print(df.head(3))
print(df.corr())
# 독립변수(x, feature) : 적절성, 종속변수(y, label or class) : 만족도.     class는 분류에서 부름. feature,label은 머신러닝에서의 명칭

import statsmodels.formula.api as smf
model = smf.ols(formula='만족도 ~ 적절성',data=df).fit()
print(model.summary())      # OLS Regression Results
print("회귀계수 : ",model.params)
print("결정계수 : ",model.rsquared)
print("p-value(유의확률값) : ",model.pvalues)
# print("예측값 : ", model.predict())

# 시각화
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.scatter(df.적절성, df.만족도)
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)        # 몇차 방정식인지 숫자를 넣어줘야함.
plt.plot(df.적절성, df.적절성 * slope + intercept)
plt.show()

# 결과 예측
print('예측값 : ', model.predict()[:5])
print('실제값 : ', df.만족도[:5].values)

# 새로운 데이터(적절성) 결과 예측
new_df = pd.DataFrame({'적절성':[4,3,1,0,6]})
new_pred = model.predict(new_df)
print(new_pred)

"""
                         OLS Regression Results                            
==============================================================================
Dep. Variable:                    만족도   R-squared:                       0.588
Model:                            OLS   Adj. R-squared:                  0.586
Method:                 Least Squares   F-statistic:                     374.0
Date:                Wed, 10 May 2023   Prob (F-statistic):           2.24e-52
Time:                        14:50:41   Log-Likelihood:                -207.44
No. Observations:                 264   AIC:                             418.9
Df Residuals:                     262   BIC:                             426.0
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.7789      0.124      6.273      0.000       0.534       1.023
적절성            0.7393      0.038     19.340      0.000       0.664       0.815
==============================================================================
Omnibus:                       11.674   Durbin-Watson:                   2.185
Prob(Omnibus):                  0.003   Jarque-Bera (JB):               16.003
Skew:                          -0.328   Prob(JB):                     0.000335
Kurtosis:                       4.012   Cond. No.                         13.4
==============================================================================
"""
# ols(Ordinary Least Square) 해석 --------------- 
"""
여기서 독립변수 '적절성' 1개. 
적절성\coef : 기울기(slope),      Intercept\coef : bias(절편).
P>|t|는 각 변수에 대한 p-value로 여기서 0.05보다 작기때문에 변수는 유의하다고 해석할 수 있다.
p-value는 t-value를 통해 얻었다. t-value가 클수록 p-value는 작아진다.

모집단평균과 표본평균이 조금은 다를 것이다. 이것은 표준편차(Intercept\stderr)
표준오차(적절성\stderr)은 작을 수록 좋다.

t는 t-value. coef(기울기) ÷ stderr(표준오차) 로 구함. - 두 변수 간의 차이

f-value는 (t-value)²
Prob (F-statistic): 2.24e-52    <<<< 모델의 유의성을 판단하는 p-value. 이 모델이 유의하면 인과관계가 있다라고 해석할 수 있다.

독립변수가 적절하지 않을때 AIC값은 큰 값을 갖게 된다. 독립변수가 적절할 수록 최저값을 가짐.

독립변수 1개 일때 : R-squared - 이 모델에 대한 설명력(결정계수).
독립변수 여러개 일때 : Adj. R-squared(수정된 R-squared) - 이 모델에 대한 설명력(결정계수).

R-squared = (상관계수)²  or  1-(ssr/sst)

Skew : 왜도 - 정규분포의 치우쳐짐을 알수있는 척도. 음수일때 왼쪽으로 꼬리가 길고 양수일때 오른쪽으로 꼬리가 길다. 0에 가까울 수록 정규분포가 평균에 가깝다는 것.
Kurtosis : 첨도 - 정규분포의 볼록함을 알 수 있는 정도. 완만할 수록 음수, 뾰족할수록 양수.

Durbin-Watson : 잔차의 독립성을 확인할 수 있는 값. 자기상관...
Jarque-Bera : 적합도 검정
Prob(JB) : 오차의 정규성. 0.05 보다 작을 수록 좋다.
Cond. No. : 다중공선성

선형회귀분석에서 귀무/대립가설
귀무 : 집단 간 차이가 없다. 기울기 = 0
대립 : 집단 간 차이가 있다. 기울기 ≠ 0

"""


