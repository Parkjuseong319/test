# 가설 검정 중 교차분석(카이제곱, 카이스퀘어, chi2)
# 카이제곱분포표를 사용 - 데이터의 분산이 퍼져있는 모습을 분포로 만든 것.
# 범주형 자료를 사용
# 일원카이제곱(변인 : 단수 - 적합도(선호도) 검정) : 교차분할표 사용 X 
# 이원카이제곱(변인 : 복수 - 독립성, 동질성 검정) : 교차분할표 사용 O
# 모집단의 모수를 추정하기 위해 샘플 데이터로 검정통계량 χ² 값을 구해 이를 사용.
# χ² = Σ(관측값 - 기댓값)² / 기댓값

import pandas as pd

# 가설 설정
# 귀무 : 벼락치기 공부하는 것과 합격여부는 관계없다.
# 대립 : 벼락치기 공부하는 것과 합격 여부는 관계가 있다.
data = pd.read_csv('../testdata/pass_cross.csv', encoding='euc-kr')
print(data.head())
print()     # 공부함 : 1, 공부안함 0, 합격 : 1, 불합격 : 0
print(data[(data['공부함'] == 1) & (data['합격'] == 1)].shape[0])    # 18
print(data[(data['공부함'] == 1) & (data['불합격'] == 1)].shape[0])   # 7
print()
ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
ctab.columns = ['합격', '불합격', '행합']
ctab.index = ['공부함', '공부안함', '열합']
print(ctab)

# 기대값(도수) = (각 행의 주변합) * (각 열의 주변합) / 총합
chi_value = (18 - 15)**2 / 15 + (7 - 10)**2/10 + (12 - 15)**2 / 15 + (13 - 10)**2/10 
print("카이제곱 : ",chi_value)      # 3.0

# 자유도 : (행개수 - 1) * (열 개수 - 1)이므로   (2-1) * (2-1) = 1
# 카이제곱표로 임계값 : 3.84 > 3.0 이므로 귀무 채택
# 평가 : 카이제곱 검정통계량 값(3.0)이 임계값(3.84)보다 작으므로 검정통계량이 귀무 채택역에 존재한다. 
# ∴ 귀무가설을 채택한다 - 벼락치기 공부하는 것과 합격여부는 관계없다.
# 위에 수집된 자료는 우연히 발생한 데이터이다.

# 평가 방법2 : p-value 사용
import scipy.stats as stats

chi2, p, _, _ = stats.chi2_contingency(ctab)    # python 카이제곱 함수는 chi2_contigency이다.
print('test statistic : {}, p-value : {}'.format(chi2, p))
# test statistic : 3.0, p-value : 0.5578254003710748    <- 서로 반비례 관계
# 평가 : 유의확률(p-value : 0.5578254003710748) > 유의수준(α, 0.05) 이므로 귀무 채택



