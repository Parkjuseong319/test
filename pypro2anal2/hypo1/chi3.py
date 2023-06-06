# 이원카이제곱 : 두 개 이상 집단이 있기 때문에 교차분할표 사용
# : 두 개 이상의 집단 또는 범주의 변인을 대상으로 동질성 or 독립성 검정.
# : 유의확률에 의해서 집단 간에 '차이가 있는가? 없는가?' 로 가설을 검정한다.

# 교육수준과 흡연률 간의 관련성 분석
import pandas as pd
import scipy.stats as stats

# 귀무 : 교육수준과 흡연률 간의 관련이 없다.(독립이다)
# 대립(연구) : 교육수준과 흡연률 간의 관련이 있다.(독립이 아니다)
data = pd.read_csv('../testdata/smoke.csv')
print(data.head(5))
print(data['education'].unique())       # [1 2 3]
print(data['smoking'].unique())         # [1 2 3]

ctab = pd.crosstab(index=data['education'], columns=data['smoking'])
ctab.index = ['대학원졸', '대졸', '고졸']
ctab.columns = ['과흡연', '보통', '노담배']
print(ctab)

chi2, p, ddof, _ = stats.chi2_contingency(ctab)
print('chi2 : {}, p : {}, ddof : {}'.format(chi2, p, ddof))
# chi2 : 18.910915739853955, p : 0.0008182572832162924, ddof : 4
# 해석 : p-value : 0.0008182 < 0.05이므로 귀무가설 기각
# ∴ 교육수준과 흡연률 간의 관련이 있다.
# 후속 조치 : ....

# Yates 보정 : 자유도가 1이라면 관측치는 0.5씩 기대값으로 옮겨 검정통계량이 더 낮게 나오게 하는 것.

