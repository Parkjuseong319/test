import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

advdf = pd.read_csv('../testdata/Advertising.csv', usecols=[1,2,3,4])
print(advdf.head(3), ' ', advdf.shape)
print(advdf.info())
print('단순 선형회귀모델 : tv, sales')
print('상관계수(r) : ', advdf.loc[:, ['sales', 'tv']].corr())   
# 0.782224 : 강한 양이 상관관계. 인과관계가 있다고 가정하고 회귀모델 작성

# lm = smf.ols(formula="sales~tv", data=advdf)
# lm = lm.fit()
lm = smf.ols(formula="sales~tv", data=advdf).fit()      # fit()은 회귀모델을 학습시키는 함수.
print(lm.summary())     # Log-Likelihood : 가늠도
print(lm.rsquared)      # 0.611875050850071

# 시각화
# plt.scatter(advdf.tv, advdf.sales)
# y_pred = lm.predict(advdf.tv)
# plt.plot(advdf.tv, y_pred, c='blue')
# plt.xlabel('TV매체 광고비')
# plt.ylabel('판매 실적')
# plt.title('선형회귀')
# plt.xlim(-50,400)    # x축 범위
# plt.ylim(ymin=0)    # y축 범위
# plt.show()

# 모델 검정
pred = lm.predict(advdf.tv[:5])     # 사실은 검정용 데이터를 따로 준비 후 사용하는 것이 옳다.
print('예측값 : ', pred.values)        # [17.97077451  9.14797405  7.85022376 14.23439457 15.62721814]
print('실제값 : ', advdf.sales[:5].values) # [22.1 10.4  9.3 18.5 12.9]


# 예측 : 새로운 tv 값으로 sales 추정
x_new = pd.DataFrame({'tv':[110.5, 220.6, 500.8]})
pred_new = lm.predict(x_new)
print("추정값 : ",pred_new)

print('다중선형회귀모델')
print(advdf.corr())
# lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data=advdf).fit()
# print(lm_mul.summary())
lm_mul = smf.ols(formula='sales ~ tv + radio', data=advdf).fit()    # newspaper의 p-value값 0.86이라 제거.
print(lm_mul.summary())

# 예측 : 새로운 tv, radio 값으로 sales 추정
x_new2 = pd.DataFrame({'tv':[110.5, 220.6, 500.8], 'radio':[10.1, 30.2, 100.1]})
pred_new2 = lm_mul.predict(x_new2)
print("추정값2 : ",pred_new2.values)

print('********' * 10)

# *** 선형회귀분석의 기존 가정 충족 조건 ***
# . 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
# . 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
# . 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
# . 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.

# 잔차항 구하기
fitted = lm_mul.predict(advdf.iloc[:, 0:2])
residual = advdf['sales'] - fitted
print('실제값 : ', advdf['sales'][:5].values)
print('예측값 : ', fitted[:5].values)
print(residual[:5].values)
print('잔차값의 평균 :',np.mean(residual))    # 1.1457501614131616e-14

import seaborn as sns
print('\n선형성(예측값과 잔차가 비슷한 패턴 유지) -----------')
sns.regplot(x=fitted, y=residual, lowess = True, line_kws={'color':'red'})  # 속성을 명시적으로 줘야 에러가 안난다.
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color= 'gray')  # 잔차의 평균은 0에 가까울수록 좋기 때문에 0을 지나는 기준선 하나를 그려준것이다.
plt.show()  # 잔차의 추세선이 파선을 기준으로 일정하지 않음. 선형성을 만족하지 않음

print('\n정규성(잔차가 정규성을 따라야함)------------')   # Jarque-Bera 값과 관련 있다.
import scipy.stats as stats
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x=x ,y=y)
plt.plot([-3,3],[-3,3],'--',color='gray')      # 그냥 하나의 기준선으로 분포가 일직선으로 한방향을 향해 가는지 확인하기 위해 그려줬다.
plt.show()      # 커브를 그리는 것이 다소 의심스럽다.

print('shapiro test : ', stats.shapiro(residual).pvalue)    #  4.190036317908152e-09 < 0.05 정규성 만족하지 못한다.

print('\n독립성(잔차가 독립성 유지, 자기상관이 있는지 확인)-----------')
# Durbin-Watson 테스트는 회귀 모델에서 잔차의 자기 상관관계에 대한 척도입니다. Durbin-Watson 테스트는 0~4의 척도를 사용하며,
# 0~2 값은 양의 자기 상관관계를 나타내며 2~4 값은 음의 자기 상관관계를 나타냅니다.
# 즉 2에 가까운 값일 수록 독립적이다.
# Durbin-Watson 검정결과 : 2.081    <<< ols에서 보여준다.
# ∴ 자기상관이 없다고 해석할 수 있다.

print("\n등분산성(잔차의 분산이 일정한지 확인)-----------------")   # 선형성과 관련이 깊다.
sns.regplot(x=fitted, y=np.sqrt(np.abs(sr)), lowess = True, line_kws={'color':'red'})   # 비모수적 최저 모델을 추정할때 lowess=True 속성 부여
plt.show()  # 평균을 기준으로 데이터 골고루 퍼지지않음. 등분산성 불만족.
# 평균선에 대해 수평적이여야 등분산성 만족. 그 외는 이분산성이라고 한다.

print('\n다중공선성(독립변수들 간의 강한 상관관계 확인)-------------------------')
# VIF란, Variance Inflation Factor의 약자로서, 분산 팽창 인수라고 합니다.
# 이 값은 다중회귀분석에서 독립변수가 다중 공선성(Multicollnearity)의 문제를 갖고 있는지 판단하는 기준이며, 
# 주로 10보다 크면 그 독립변수는 다중공선성이 있다고 말합니다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(variance_inflation_factor(advdf.values, 1))   # tv    12.57031
print(variance_inflation_factor(advdf.values, 2))   # radio 3.153498

print()
# 참고 : Cook's distance - outlier 확인 지표
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm_mul).cooks_distance   # outlier값 반환
print(cd.sort_values(ascending=False).head())

import statsmodels.api as sm
sm.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()

print(advdf.iloc[[130, 5, 35, 178, 126]])





