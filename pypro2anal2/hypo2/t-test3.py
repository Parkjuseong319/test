# 어느 음식점 매출 데이터, 날씨 데이터를 활용하여 강수 여부에 따른 매출의 평균 차이를 검정
# 귀무 : 강수 여부에 따른 매출액의 평균 차이가 없다.
# 대립 : 강수 여부에 따른 매출액의 평균 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출 데이터 읽기
sales_data = pd.read_csv('../testdata/tsales.csv', dtype={'YMD':'object'})  # wt_data['tm'] 과 데이터 타입을 맞추기 위해 타입변경
print(sales_data.head(3))
print(sales_data.info())

# 날씨 데이터 읽기
wt_data = pd.read_csv('../testdata/tweather.csv')
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-',''))
print(wt_data.head(3))
print(wt_data.info())
print()
# 두 데이터를 병합(merge)
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm')
print(frame.head(3))
print(frame.columns)

data = frame.iloc[:,[0,1,7,8]]      # YMD, AMT, maxTa, sumRn column. maxTa 칼럼은 ANOVA 때문에 꺼냄
print(data.head(3))
print(data.isnull().sum())

print('-------- t-test----------')
# print(data['sumRn'] > 0)
data['rain_yn'] = (data['sumRn'] > 0).astype(int)
# data['rain_yn'] = (data.loc[:,('sumRn')] > 0) * 1
print(data.head(3))

# 강수량 있는 집단, 강수량 없는 집단으로 분리
import matplotlib.pyplot as plt

sp = np.array(data.iloc[:, [1, 4]])     # AMT, rain_yn column. 2차원 배열로 가공
# print(sp)
tg1 = sp[sp[:,1] == 0, 0]       # 집단1 : 비가 안왔을 때 집단의 매출액
tg2 = sp[sp[:,1] == 1, 0]       # 집단2 : 비가 왔을 때 집단의 매출액
# print(tg1[:3])
# print(tg2[:3])
# plt.plot(tg1)
# plt.show()
# plt.plot(tg2)
# plt.show()
plt.boxplot([tg1, tg2], notch=True, meanline=True, showmeans=True)      # meanline은 평균선, notch는 중위수, showmeans는 평균값 보이기
plt.show()

print(np.mean(tg1))         # 761040.2542372881
print(np.mean(tg2))         # 757331.5217391305
print('매출액 평균 차이 : ', 761040.2542372881 - 757331.5217391305)    # 3708.732498157653

# 정규성 확인    ( N > 30 )
print(len(tg1), len(tg2))   # 236 92
print(stats.shapiro(tg1).pvalue)        # 0.056049469858407974
print(stats.shapiro(tg2).pvalue)        # 0.882739782333374

# 등분산성 확인
print(stats.levene(tg1, tg2).pvalue)    # 0.7123452333011173

# 모든 p-value값이 0.05보다 크므로 정규성, 등분산성 만족한다.

print(stats.ttest_ind(tg1, tg2, equal_var=True))
# Ttest_indResult(statistic=0.10109828602924716, pvalue=0.919534587722196)
# 해석 : 정규성, 등분산성은 조건을 충족 
# p-value=0.919534587722196 > 0.05 이므로 귀무 채택. 대립 기각
# ∴ 강수 여부에 따른 매출액의 평균 차이가 없다.

