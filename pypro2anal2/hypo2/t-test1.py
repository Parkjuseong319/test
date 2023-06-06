# T-test
# 집단 간 차이분석: 평균 또는 비율 차이를 분석
# 모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.
# 독립변수 : 범주형, 종속변수 : 연속형
# t분포 사용 : 표본 평균을 이용하여 정규분포의 평균을 해석할 때 많이 사용
# 두 개 이하의 집단 비교를 위한 평균과 표준편차의 비율 차이를 구하고 그 차이가 우연인지 아닌지를 판단하는 검정방법.

# * 단일 모집단의 평균에 대한 가설검정(one samples t-test)
# one samples t-test는 정규분포에 대해 기댓값을 조사하는 검정방법

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 실습 예제 1)
# 어느 남성 집단의 평균 키 검정.
# 하나의 집단에 대한 표본평균이 예측된 평균과 같은지 여부를 확인 
# 귀무 : 집단의 평균 키가 177이 이다.     
# H0 : 표본의 평균(x̄) - 모집단평균(μ),  표본의 평균 - 모집단 평균 = 0
# 대립 : 집단의 평균 키가 177이 아니다.   
# H1 : 표본의 평균(x̄) ≠ 모집단평균(μ), 표본의 평균 - 모집단 평균 > 0 or 표본의 평균 - 모집단 평균 < 0

one_sample = [167.0, 162.9, 169.2, 176.7, 187.5]
print(np.mean(one_sample))      # 172.66  
result = stats.ttest_1samp(one_sample, popmean=177)
print('statistic : %.3f, pvalue:%.3f'%result)   # TtestResult(statistic=-1.0011966911793568, pvalue=0.3733875041602407, df=4)
# statistic : -1.001, pvalue:0.373
# 해석 : p-value : 0.373 > 0.05 이므로 귀무 채택. 대립 기각
print()
result2 = stats.ttest_1samp(one_sample, popmean=185)
print('statistic : %.3f, pvalue:%.3f'%result2)
# statistic : -2.847, pvalue:0.047
# 해석 : p-value : 0.047 < 0.05 이므로 대립 채택, 귀무기각.

print("--------"*10)
# 실습 예제 2)
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균검정) student.csv
# 귀무 : A중학교 1학년 1반 학생들의 국어 점수 평균은 80 이다.
# 대립 : A중학교 1학년 1반 학생들의 국어 점수 평균은 80이 아니다.

data = pd.read_csv('../testdata/student.csv')
print(data.head(2))
print(data['국어'].mean())    # 72.9
result3 = stats.ttest_1samp(data.국어, popmean=80)
print('statistic : %.3f, pvalue:%.3f'%result3)
# statistic : -1.332, pvalue:0.199
# 해석 : p-value : 0.199 > 0.05 이므로 귀무 채택
# ∴ A중학교 1학년 1반 학생들의 국어 점수 평균은 80 이다.
# student.csv 문서 자료는 우연히 발생한 데이터들이다. 






