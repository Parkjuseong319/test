import pickle
import MySQLdb
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
#   남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
#   여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4
# 귀무 : 남자와 여자 그룹의 혈관 내의 콜레스테롤 양에 차이가 없다.
# 대립 : 남자와 여자 그룹의 혈관 내의 콜레스테롤 양에 차이가 있다.

male = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
female = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6 ,7.7 ,8.8, 6.6, 6.4]
# 비복원 추출
rmale = random.sample(male, k=15)         # random.choice(male, size=15(개수), replace=True(중복허용) or False(중복불허)) 
rfemale = random.sample(female, k=15)

# 정규성 확인
print(stats.shapiro(rmale).pvalue)       # 0.013721521943807602
print(stats.shapiro(rfemale).pvalue)     # 0.011043859645724297

# 등분산성 확인
print(stats.bartlett(rmale, rfemale).pvalue) # 0.6444069767964012
# 정규성 만족 X, 등분산성 만족 O. 같은 개수를 비복원 추출 하기 때문에 wilcoxon 함수 사용
result1 = stats.wilcoxon(rmale, rfemale)

print(result1)      # statistic=42.0, pvalue=0.33026123046875
# 해석 : pvalue=0.8454793961047851 > 0.05 이므로 귀무 채택. 대립 기각
# ∴ 남자와 여자 그룹의 혈관 내의 콜레스테롤 양에 차이가 없다.

#---------------------------------------------------------------------------
 
# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.
# 귀무 : 총무부, 영업부 직원의 연봉 평균은 차이가 없다.
# 대립 : 총무부, 영업부 직원의 연봉 평균은 차이가 있다.
try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
    conn = MySQLdb.connect(**config)
except Exception as e:
    print('DB connect error', e)
    
try:    
    # 총무부
    sql1 = 'select jikwon_pay from jikwon join buser on jikwon.buser_num=buser.buser_no where buser_name ="총무부"'
    df1 = pd.read_sql(sql1, conn)
    
    # 영업부
    sql2 = 'select jikwon_pay from jikwon join buser on jikwon.buser_num=buser.buser_no where buser_name ="영업부"'
    df2 = pd.read_sql(sql2, conn)
    print(df1.mean())   # 5414.285714
    print(df2.mean())   # 4908.333333
    
    # 연봉 없는 직원 평균 연봉으로 채우기
    re_df1 = df1.fillna(df1.mean()) 
    re_df2 = df2.fillna(df2.mean()) 
    
    # 정규성 확인
    print(stats.shapiro(re_df1).pvalue)       # 0.02604489028453827
    print(stats.shapiro(re_df2).pvalue)         # 0.025608452036976814
    
    # 등분산성 확인. 1차원 데이터로 변환
    cpay = []
    ypay = []
    for i in re_df1['jikwon_pay']:
        cpay.append(i)
    for i in re_df2['jikwon_pay']:
        ypay.append(i)
    
    print(stats.levene(cpay, ypay).pvalue)      # 0.915044305043978
    # 정규성 불만족, 등분산성 만족
    result2 = stats.mannwhitneyu(cpay, ypay)
    print(result2)      # (statistic=51.0, pvalue=0.47213346080125185)
    # 해석 : p-value=0.47213346080125185 > 0.05 이므로 귀무 채택, 대립 기각
    # ∴ 총무부, 영업부 직원의 연봉 평균은 차이가 없다.

except Exception as e:
    conn.close()

#---------------------------------------------------------

# [대응표본 t 검정 : 문제4]
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 유지되고 있다고 말하고 있다.
# 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 점수는 학생 번호 순으로 배열되어 있다.
#    중간 : 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80
#    기말 : 90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95
# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?
# 귀무 : 이 학급의 학업능력이 변화했다고 이야기 할 수 없다.
# 대립 : 이 학급의 학업능력이 변화했다고 이야기 할 수 있다. 
mid = [80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
last = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]

# 정규성 확인
print(stats.shapiro(mid).pvalue)     # 0.368144154548645     
print(stats.shapiro(last).pvalue)   # 0.19300280511379242

# 정규성 만족.

sample = stats.ttest_rel(mid, last)
print(sample) # statistic=-2.6281127723493998, pvalue=0.023486192540203194, df=11
# 해석 : p-value=0.023486 < 0.05 이므로 귀무 기각, 대립 채택
# ∴ 이 학급의 학업능력이 변화했다고 이야기 할 수 있다.








