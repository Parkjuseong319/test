# 두 집단 간 차이분석 : 평균 또는 비율 차이를 분석
# 독립 표본 t검정(independent two sample t-test)
# 선행 조건 : 두 집단은 정규분포를 따라야 한다. 두 집단의 분산이 동일하다는 가정이 필요.
import numpy as np
import pandas as pd
from scipy import stats

print((179.8-178.5) / (6.05 / np.sqrt(101)))        # 2.1594

# df : 100, cv : 1.984 이므로 t-value는 임계치 밖에 있다.
# ∴ 두 집단간 평균의 차이가 있다. 

# 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.

# 실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정
# 남녀 두 집단 간 파이썬 시험의 평균 차이가 약 11.5점 정도 나는데 우연히 발생할 확률은 얼마인가?
# 만약 우연히 발생했다면 평균은 같은 것이고, 우연히 발생하지 않았다면 다른 것이다.
# 귀무 : 남녀 두 집단 간 파이썬 시험의 평균 차이가 없다.
# 대립 : 남녀 두 집단 간 파이썬 시험의 평균 차이가 있다.
male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]
print(np.mean(male), np.mean(female))   # 83.8 72.24

# two_sample = stats.ttest_ind(male, female)
two_sample = stats.ttest_ind(male, female, equal_var = True)    # 등분산성이 같다 라고 가정.(기본값임)
print(two_sample)       # statistic=1.233193127514512, pvalue=0.2525076844853278
# 결론 : p-value(0.25250) > 0.05 이므로 귀무 채택. 대립 기각
# ∴ 남녀 두 집단 간 파이썬 시험의 평균 차이가 없다. 즉, 우연히 발생한 데이터.

print('------'*8)
# 실습) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv
data = pd.read_csv('../testdata/two_sample.csv')
print(data.head(3))
print(data.isnull().sum())      # NaN 2개

ms = data[['method', 'score']]
print(ms)
m1 = ms[ms['method'] == 1]
m2 = ms[ms['method'] == 2]

score1 = m1['score']
score2 = m2['score']
print(score1)
print(score2)
sco1 = score1.fillna(score1.mean())     # 결측치 평균으로 채우는 것이 합리적. 0으로 채우면 극단적이다.
sco2 = score2.fillna(score2.mean())
# print(sco1)
# print(sco2)
print(sco1.isnull().sum())
print(sco2.isnull().sum())

result = stats.ttest_ind(sco1, sco2)
print('t-value:%.5f, p-value:%.5f'%result)  # t-value:-0.19649, p-value:0.84505

# 선행 조건1 : 두 집단은 정규분포를 따라야 한다.
# 시각화로 정규성 확인
import matplotlib.pyplot as plt
import seaborn as sns
# sns.histplot(sco1, kde= True)
# sns.histplot(sco2, kde= True)   # kde 밀도 추정곡선
# plt.show()

# 정규성 확인 함수 : p-value > 0.05 보다 크면 정규성 만족
print(stats.shapiro(sco1).pvalue)   # p-value : 0.36798644065856934
print(stats.shapiro(sco2).pvalue)   # p-value : 0.6714232563972473

# 선행 조건2 : 두 집단의 분산이 동일하다는 가정이 필요. p-value > 0.05 보다 크면 등분산성 만족
print(stats.levene(sco1, sco2).pvalue)      # 모수 검정. 일반적으로 많이 사용    0.4568
print(stats.fligner(sco1, sco2).pvalue)     # 모수 검정                     0.4432
print(stats.bartlett(sco1, sco2).pvalue)    # 비모수 검정(데이터 개수 30개 미만일때) 0.2679

# 정규성, 등분산성 만족하므로 ttest_ind() 사용
result = stats.ttest_ind(sco1, sco2, equal_var = True)  # 등분산성 만족
# result = stats.ttest_ind(sco1, sco2, equal_var = False)  # 등분산성 불만족
print('t-value:%.5f, p-value:%.5f'%result) 

# if 정규성을 만족 못할때 
# result2 = stats.wilcoxon(sco1, sco2)        # 두 집단의 샘플의 크기가 같은 경우
# print('t-value:%.5f, p-value:%.5f'%result2) 
result2 = stats.mannwhitneyu(sco1, sco2)      # 두 집단의 샘플의 크기가 다른 경우
print('t-value:%.5f, p-value:%.5f'%result2) 


