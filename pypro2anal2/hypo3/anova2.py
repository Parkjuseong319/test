# 일원분산분석으로 집단 간 평균차이 검정
# 강남구에 있는 gs25(요인) 3개지역(집단) 알바생의 급여에 대한 평균의 차이가 있는지 검정
# 귀무 : gs25 3개지역 알바생 급여 평균은 차이가 없다.
# 대립 : gs25 3개지역 알바생 급여 평균은 차이가 있다.
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# data = pd.read_csv('../testdata/group3.txt', header=None)    # dataframe으로 읽기
# print(data)

data = np.genfromtxt('../testdata/group3.txt', delimiter=',')   # matrix로 읽기
print(data[:2])

# 세 개의 집단에 대한 급여, 평균
gr1 = data[data[:,1]==1, 0]
gr2 = data[data[:,1]==2, 0]
gr3 = data[data[:,1]==3, 0]
print('gr1 : ', gr1, ' ', np.mean(gr1))     # 316.625
print('gr2 : ', gr2, ' ', np.mean(gr2))     # 256.44444444444446
print('gr3 : ', gr3, ' ', np.mean(gr3))     # 278.0

# 정규성
print(stats.shapiro(gr1).pvalue)    # 1개만 볼때 shapiro() 사용. 0.33368 > 0.05 만족
print(stats.shapiro(gr2).pvalue)    # 0.65610 > 0.05 만족
print(stats.shapiro(gr3).pvalue)    # 0.83248 > 0.05 만족

# 등분산성
print(stats.levene(gr1,gr2,gr3))    # p-value=0.04584681 
print(stats.bartlett(gr1,gr2,gr3))  # p-value=0.35080326 > 0.05 만족

# 데이터 퍼짐 정도
# plt.boxplot([gr1,gr2,gr3], showmeans=True)
# plt.show()

# 일원분산분석 방법1 : anova_lm()
df = pd.DataFrame(data, columns=['pay', 'group'])
# print(df)
lmodel = ols('pay~C(group)', data=df).fit()
print(anova_lm(lmodel, type=1))         # p-value : 0.043589 < 0.05이므로 귀무 기각, 대립채택

# 일원분산분석 방법2 : f_oneway()
f_statistic, pvalue = stats.f_oneway(gr1,gr2,gr3)
print('f_statistic:{}, p-value:{}'.format(f_statistic, pvalue)) # p-value:0.043589

# 결론 : p-value:0.0435893349 < 0.05 이므로 귀무 기각, 대립 채택
# ∴ gs25 3개지역 알바생 급여 평균은 차이가 있다.

print('사후 검정 --------------------')
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_result = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tukey_result)
# group1 에서 차이가 있다고 분석된다.


# tukey_result를 시각화
tukey_result.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show()

