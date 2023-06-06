import numpy as np
import scipy.stats as stats
import pandas as pd
# 1:여아, 2:남아
# 귀무 : 남아 신생아 몸무게 평균은 3000g 이다.
# 대립 : 남아 신생아 몸무게 평균은 3000g이 아니다.

# df = pd.read_csv('../testdata/babyboom.csv')
# baby = df[df['gender']==2]['weight']
# result = stats.ttest_1samp(baby, popmean=3000)
# print(result)
# statistic=4.47078356044109, pvalue=0.00014690296107439875, df=25
# 대립 채택. 

# x = [1,2,3,4,5]
# y = [8,7,6,4,5]
# print(np.corrcoef(x,y))


# 귀무 : 매출액은 포장지 색상과는 관련이 없다.
# 대립 : 매출액은 포장지 색상과는 관련이 있다.

"""
blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

print(np.mean(blue), ' ', np.mean(red))     # 72.81  63.81

# 정규성 확인(만족)
print(stats.shapiro(blue).pvalue)   # 0.5102316737174988
print(stats.shapiro(red).pvalue)    # 0.5347929000854492

# 등분산성 확인(데이터 30개 미만)
print(stats.bartlett(blue, red).pvalue)     # 0.3626749441773801

# 정규성 등분산성 p-value값이 모두 0.05 이상이므로 만족
result = stats.ttest_ind(blue, red, equal_var = True)
print('t-value:%.5f, p-value:%.5f'%result)
# t-value:2.92802, p-value:0.00832
"""
"""
# 키보드로 국어 점수를 입력하면 수학 점수 예측 (80점을 입력하자)

import statsmodels.formula.api as smf

student = pd.read_csv('../testdata/student.csv')
print(student.head(5))
print(student.corr())

#- 국어 점수 입력하면 수학 점수 예측
model = smf.ols('수학~국어',data=student).fit()
print(model.summary())
print(0.5705 * 80 + 32.1069)
print(model.predict(pd.DataFrame({'국어':[80]})))
kor = int(input('국어점수를 입력하시오'))
print("당신의 대략의 수학점수 : ", np.round(model.predict(pd.DataFrame({'국어':[kor]})).values[0],2))

"""
