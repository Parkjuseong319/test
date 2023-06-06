# 카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
  # 예제파일 : cleanDescriptive.csv
  # 칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
  # 조건 :  level, pass에 대해 NA가 있는 행은 제외한다.

# 귀무 : 부모 학력수준과 자녀의 진학여부는 관련이 없다.
# 대립 : 부모 학력수준이 자녀의 진학여부와 관련이 있다.
import pandas as pd
import scipy.stats as stats
data1 = pd.read_csv('../testdata/cleanDescriptive.csv').dropna(subset=['level', 'pass'])
print(data1)
print(data1['level'].unique())      # [1. 2. 3.]
print(data1['pass'].unique())       # [2. 1.]

ctab = pd.crosstab(index=data1['level'], columns=data1['pass'])
ctab.columns=['합격', '불합격']
ctab.index = ['고졸', '대졸', '대학원']
print(ctab)

chi2, p, df, _ = stats.chi2_contingency(ctab)
print(chi2, p, df)
# 해석1 : p-value : 0.250705684 > 0.05 이므로 귀무가설 채택, 대립가설 기각
# 해석2 : df(2), chi2(2.767), p-value(0.251), cv(5.99)
# chi2값이 cv 내에 있으므로 귀무가설 채택역에 존재. 귀무가설 채택.
# ∴ 부모학력 수준이 자녀의 대학진학 여부와는 관련이 없다.


# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
# 그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
#   예제파일 : MariaDB의 jikwon table 
#   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#   jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#   조건 : NA가 있는 행은 제외한다.

# 귀무 : 직급과 연봉은 관련이 없다.
# 대립 : 직급과 연봉은 관련이 있다.
import numpy as np
import pickle
import MySQLdb
with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)
conn = MySQLdb.connect(**config)
data2 = pd.read_sql('select jikwon_jik, jikwon_pay from jikwon', conn)

dfcut1 = pd.cut(data2['jikwon_pay'], bins=[1000,3000,5000,7000, np.inf],right=False ,labels=[1,2,3,4])
data2['jikwon_pay'] = dfcut1
jik = {
        '이사':1, '부장':2, '과장':3, '대리':4, '사원':5
    }
data2['jikwon_jik'] = data2['jikwon_jik'].apply(lambda x: jik[x])
# map 키의 값이 없으면 NaN 값으로 반환 
# replace 키의 값이 없으면 기존 값으로 반환
re_df = data2.dropna()
print(re_df)

ctab2 = pd.crosstab(index=re_df['jikwon_jik'], columns=re_df['jikwon_pay'])
print(ctab2)
chi2, p, df, _= stats.chi2_contingency(ctab2)
print(chi2, p, df)
# 해석 : p-value : 0.000192 < 0.05 이므로 대립가설 채택, 귀무가설 기각
# 해석2 : chi2(37.4034), df(12), cv(21.03)
# chi2 값이 cv값 내에 없으므로 귀무가설 기각, 대립가설 채택
# ∴ 직급과 연봉은 관련이 있다.


