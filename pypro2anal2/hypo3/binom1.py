# 이항분포 검정 : 결과가 두 가지 값을 가지는 확률변수의 분포를 판단하는데 효과적
# binom_test(x=성공 또는 실패 횟수, N=시도횟수, p=가설확률값)

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('../testdata/one_sample.csv')
print(data)
ctab = pd.crosstab(index=data['survey'], columns='count')
ctab.index = ['불만족', '만족']
print(ctab)

# 귀무 : 직원을 대상으로 고객 대응 서비스 교육 후 고객안내 서비스 만족률이 80%다. 
# 대립 : 직원을 대상으로 고객 대응 서비스 교육 후 고객안내 서비스 만족률이 80%가 아니다. 

print('양측 검정(방향성 X) : 기존 80% 만족률 기준 검정 실시')
x = stats.binom_test([136,14], p=0.8, alternative='two-sided')
print(x)        # 0.00067347
# 해석 : p-value : 0.00067347 < 0.05 이므로 대립채택, 귀무 기각
# ∴ 직원을 대상으로 고객 대응 서비스 교육 후 고객안내 서비스 만족률이 80%가 아니다.
# 기존 만족률보다 크다, 작다라는 방향성은 제시하지 않는다.(양측검정이라 방향성이 없기 때문)

print()
print('양측 검정(방향성 X) : 기존 80% 만족률 기준 검정 실시')
x = stats.binom_test([14, 136], p=0.2, alternative='two-sided')
print(x)        # 0.00067347
# 해석 : p-value : 0.00067347 < 0.05 이므로 대립채택, 귀무 기각
# ∴ 직원을 대상으로 고객 대응 서비스 교육 후 고객안내 서비스 만족률이 80%가 아니다.
# 기존 만족률보다 크다, 작다라는 방향성은 제시하지 않는다.(양측검정이라 방향성이 없기 때문)

print()
print('단측 검정(방향성 O) : 기존 80% 만족률 기준 검정 실시')
x = stats.binom_test([136, 14], p=0.8, alternative='greater')       # greater는 크다, less는 작다라는 방향성 제시
# x = stats.binom_test([14, 136], p=0.2, alternative='less')     # 이것의 결과도 같은 결과가 나온다.
print(x)        # 0.00031794
# 해석 : p-value : 0.00031794 < 0.05 이므로 대립채택, 귀무 기각
# ∴ 직원을 대상으로 고객 대응 서비스 교육 후 고객안내 서비스 만족률이 80%가 아니다.
# 기존 만족률보다 20% 이하로 불만율이 낮아졌다라고 할 수 있다.




