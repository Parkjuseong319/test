# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
# 참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  
import numpy as np
import pandas as pd
from io import StringIO
from scipy import stats

data = StringIO("""
구분,지상파,종편,운동
1,0.9,0.7,4.2
2,1.2,1.0,3.8
3,1.2,1.3,3.5
4,1.9,2.0,4.0
5,3.3,3.9,2.5
6,4.1,3.9,2.0
7,5.8,4.1,1.3
8,2.8,2.1,2.4
9,3.8,3.1,1.3
10,4.8,3.1,35.0
11,NaN,3.5,4.0
12,0.9,0.7,4.2
13,3.0,2.0,1.8
14,2.2,1.5,3.5
15,2.0,2.0,3.5
""")
df = pd.read_csv(data)
# print(df['지상파'].mean())     # 2.707142857
df["지상파"] = df["지상파"].fillna(df["지상파"].mean())
df.drop(df[df['운동']>10].index,axis=0, inplace=True)
print(df)

#  - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
x1 = df['지상파']
y1 = df['운동']
print(np.corrcoef(x1, y1))

re1 = stats.linregress(x1, y1)
time1 = float(input("지상파 방송 시청 시간을 입력해주세요. ex) 1.0 , 2.5..."))
print('운동시간 예측 값 : ', np.polyval([re1.slope, re1.intercept], time1))

#  - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
x2 = df['지상파']
y2 = df['종편']
print(np.corrcoef(x2, y2))

re2 = stats.linregress(x2, y2)
time2 = float(input("지상파 방송 시청 시간을 입력해주세요. ex) 1.0 , 2.5..."))
print('종편 방송 시청 시간 예측 값 : ', np.polyval([re2.slope, re2.intercept], time2))

