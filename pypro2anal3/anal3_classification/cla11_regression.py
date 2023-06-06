# 데이터명 : Boston Housing Price (보스턴 주택 가격 데이터)
# 레코드수 : 506 개
# 필드개수 :  14 개

# [01]  CRIM    자치시(town) 별 1인당 범죄율
# [02]  ZN    25,000 평방피트를 초과하는 거주지역의 비율
# [03]  INDUS    비소매상업지역이 점유하고 있는 토지의 비율
# [04]  CHAS    찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0)
# [05]  NOX    10ppm 당 농축 일산화질소
# [06]  RM    주택 1가구당 평균 방의 개수
# [07]  AGE    1940년 이전에 건축된 소유주택의 비율
# [08]  DIS    5개의 보스턴 직업센터까지의 접근성 지수
# [09]  RAD    방사형 도로까지의 접근성 지수
# [10]  TAX    10,000 달러 당 재산세율
# [11]  PTRATIO    자치시(town)별 학생/교사 비율
# [12]  B    1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함.
# [13]  LSTAT    모집단의 하위계층의 비율(%)
# [14]  MEDV    본인 소유의 주택가격(중앙값) (단위: $1,000)

# DecisionTreeRegressor, RandomForestRegressor 사용하여 정량 예측

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


boston = pd.read_csv('../testdata/housing.data', header=None, sep='\s+')
boston.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
print(boston.head(4))
boston = boston[['MEDV', 'LSTAT']]
print(boston.head(3))

print(boston.corr())
# sns.pairplot(boston[['MEDV', 'LSTAT']])
# plt.show()

# 단순 선형회귀
x = boston[['LSTAT']].values
y = boston['MEDV'].values
print(x[:3])
print(y[:3])

print('DecisionTreeRegressor -------')
model1 = DecisionTreeRegressor(criterion='friedman_mse', random_state=123).fit(x,y)
print('예측값 : ', model1.predict(x)[:10])
print('실제값 : ', y[:10])
print('결정계수 : ', r2_score(y, model1.predict(x)))

print('RandomForestRegressor -------')
model1 = RandomForestRegressor(criterion='friedman_mse',n_estimators=100, random_state=123).fit(x,y)
print('예측값 : ', model1.predict(x)[:10])
print('실제값 : ', y[:10])
print('결정계수 : ', r2_score(y, model1.predict(x)))


