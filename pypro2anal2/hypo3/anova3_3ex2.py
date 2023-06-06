import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from bokeh.layouts import column
import MySQLdb
import pickle

# [ANOVA 예제 2]
# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.
# 귀무 : 부서별 직원의 연봉 차이가 없다.
# 대립 : 부서멸 직원의 연봉 차이가 있다.
# db 연결: pickle
try:
    with open("mydb.dat", mode="rb") as obj:
        config=pickle.load(obj)
except Exception as e:
    print('DB 연결 오류: ', e)

# db 처리
try:
    conn=MySQLdb.connect(**config)
    sql="select jikwon_pay , buser_name from jikwon join buser on buser_num = buser_no"
    
    jikData=pd.read_sql(sql, conn).dropna()
    
    총무부 = jikData[jikData['buser_name'] == '총무부']['jikwon_pay']
    영업부 = jikData[jikData['buser_name'] == '영업부']['jikwon_pay']
    전산부 = jikData[jikData['buser_name'] == '전산부']['jikwon_pay']
    관리부 = jikData[jikData['buser_name'] == '관리부']['jikwon_pay']
    
    print(stats.levene(총무부, 영업부, 전산부, 관리부).pvalue) # 0.7980 > 0.05 이므로 등분산성 만족
    print(stats.shapiro(총무부).pvalue) #0.02604
    print(stats.shapiro(영업부).pvalue) #0.02560
    print(stats.shapiro(전산부).pvalue) #0.41940
    print(stats.shapiro(관리부).pvalue) #0.90780 정규성 만족 X
    
    print(stats.f_oneway(총무부,영업부,전산부,관리부)) #0.7454 > 0.05 귀무채택
    
    print(stats.kruskal(총무부,영업부,전산부,관리부)) # pvalue=0.6433
    # pvalue > 0.05 귀무채택
    
    #사후 분석
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey_result = pairwise_tukeyhsd(endog=jikData.jikwon_pay, groups=jikData.buser_name,alpha=0.05) #유의수준 적기
    print(tukey_result)
    
    # ===========================================================
    # group1 group2  meandiff  p-adj    lower      upper   reject
    # -----------------------------------------------------------
    #    관리부    영업부 -1354.1667 0.6937 -4736.5568 2028.2234  False #영향이 없다
    #    관리부    전산부  -933.9286  0.897 -4605.9199 2738.0628  False
    #    관리부    총무부  -848.2143 0.9202 -4520.2056 2823.7771  False
    #    영업부    전산부   420.2381 0.9756 -2366.0209 3206.4971  False
    #    영업부    총무부   505.9524 0.9588 -2280.3066 3292.2114  False
    #    전산부    총무부    85.7143 0.9998 -3045.7705  3217.199  False
    # -----------------------------------------------------------
    # 귀무 : 부서별 직원의 연봉 차이가 없다.
    
except Exception as e:
    print('DB 처리 오류: ', e)