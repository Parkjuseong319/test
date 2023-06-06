# 세 개 이상의 모집단에 대한 가설검정 – 분산분석(ANOVA, 변량분석)
# ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인
# 에 의한 분산이 의미 있는 크기를 크기를 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 제1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of VAriance)을 이용하게 된다.
# f-value = 그룹 간 분산(between variance) / 그룹 내 분산(within variance)
# f분포 : 서로 다른 두 개 이상의 모집단의 분산이 같은지를 확인할 때 사용한다.

# 독립변수 : 범주형, 종속변수 : 연속형

# 서로 독립인 세 집단의 평균 차이 검정 (일원분산분석:one-way ANOVA)
# 측정값에 영향을 미치는 요인(독립변수)이 1개 : 1개의 요인에 대해 세 개 이상의 집단 평균차이 검정
# 집단 간 분산이 집단 내 분산보다 충분히 큰 것인가를 파악

# 실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시. three_sample.csv
# 귀무 : 세 가지 교육방법에 따른 시험점수에는 차이가 없다. 
# 대립 : 세 가지 교육방법에 따른 시험점수에는 차이가 있다.

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols         # ols는 최소제곱법하는 함수
# statsmodels 라이브러리는 추정 및 검정, 회귀분석(시계열 분석) 등의 기능을 제공

data = pd.read_csv('../testdata/three_sample.csv')
print(data.head(3), len(data))  # 80
print(data.describe())          # score column에서 max 500이 찍히는데 이것은 outlier(이상치)

import matplotlib.pyplot as plt
# plt.hist(data.score)
# plt.boxplot(data.score)
# plt.show()

data = data.query('score <= 100')
print(data.describe())
print(len(data))        # 78
# plt.boxplot(data.score)
# plt.show()

# 분산분석 가정 만족 확인
# 등분산성 만족 확인 : 만족(ANOVA), 불만족(welch-ANOVA)
result = data[['method','score']]
m1 = result[result['method']==1]
m2 = result[result['method']==2]
m3 = result[result['method']==3]
# print(m1)
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']
print('등분산성 확인 : ', stats.levene(score1, score2, score3).pvalue)    # 0.1132285 > 0.05 등분산성 만족.
# 참고 : 등분산성 불만족시
# 처치1 : 데이터 정규화(normalization) 추천
# 처치2 : 데이터 표준화(standardization)
# 처치3 : transformation - 데이터에 자연 log를 적용
# ...

# 정규성 만족 확인 : 만족(ANOVA), 불만족(kolmogorov-Smirnov test, Kruskal-Wallis test)
print('정규성 확인 : ', stats.ks_2samp(score1, score2).pvalue)   # 0.30968796 > 0.05 만족
print('정규성 확인 : ', stats.ks_2samp(score1, score3).pvalue)   # 0.7162094  > 0.05 만족
print('정규성 확인 : ', stats.ks_2samp(score2, score3).pvalue)   # 0.77240817 > 0.05 만족

data2 = pd.crosstab(data['method'], data['survey'])
data2.index = ['방법1', '방법2','방법3']
data2.columns = ['만족','불만족']
print(data2)

import statsmodels.api as sm
# 회귀분석 시 독립변수가 범주형임을 명시적으로 기술(C(독립변수))
reg = ols('score~C(method)', data).fit()       # 선형회귀 분석 모델 생성
# reg = ols("data['score'] ~ data['method']", data=data).fit()
table = sm.stats.anova_lm(reg, type=1)
# Residual(오차) 에서 sum_sq = sse , mean_sq = mse
# C(독립변수)에서 sum_sq = ssr , mean_sq = msr 
# msr = ssr / df
# mse = sse / df
# f-value = msr/mse
# p-value는 f값으로 얻음.
# sst = sse(흔히 잔차라고 함) + ssr(예측값-평균)
# https://acdongpgm.tistory.com/70

print(table)    # p-value :  0.939639 > 0.05 이므로 귀무 채택.
# ∴ 세 가지 교육방법에 따른 시험점수에는 차이가 없다.

print('독립변수 복수')
reg2 = ols('score~C(method + survey)', data).fit()
table2 = sm.stats.anova_lm(reg2, type=1)
print(table2)

# 사후검정(분석) - Post Hoc Test : Tukey, Dunnett, Sheffe...
# ANOVA 분석 결과가 통계적으로 유의하다는 결과를 얻었을 경우 그것은 집단별로 차이가 있다는 것까지는 도출가능하지만, 
# 어떤 집단간에 차이가 있는지는 알려주지 않습니다.
# N개의 집단 중 어떤집단들간에 값이 차이가 있는지를 추가적으로 살펴보기 위해서 실시하는것이 사후 분석
import numpy as np
print(np.mean(score1), ' ',np.mean(score2), ' ',np.mean(score3))
# 67.38461538461539   68.35714285714286   68.875

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_result = pairwise_tukeyhsd(endog=data.score, groups=data.method)
print(tukey_result)
# tukey_result reject컬럼에서 True는 차이가 있다, False는 차이가 없다 라고 해석 가능하다.

# tukey_result를 시각화
tukey_result.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show()



