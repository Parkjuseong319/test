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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv('../testdata/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
print(df.head(2), df.shape)     # (506, 14)
print(df.corr())    # LSTAT(독립변수), MEDV(종속변수) -0.737663  < 둘다 연속형데이터

x = df[['LSTAT']].values
y = df['MEDV'].values
print(x[:2])
print(y[:2])
model = LinearRegression()

# 다항특성
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)
print(x_quad[:2])
print(x_cubic[:2])

# 단순회귀
model.fit(x,y)
x_fit = np.arange(x.min(), x.max(),1)[:, np.newaxis]    # 그래프 표시용
y_lin_fit = model.predict(x_fit)
# print(y_lin_fit)
model_r2 = r2_score(y,model.predict(x))
print(model_r2)     # 0.544146

 # degree = 2
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))   # 다항회귀
q_r2 = r2_score(y,model.predict(x_quad))
print(q_r2)     # 0.640716

# degree = 3
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))   # 다항회귀
c_r2 = r2_score(y,model.predict(x_cubic))
print(c_r2)     # 0.6578476

# 시각화
plt.scatter(x, y, c='lightgray', label='학습데이터')
plt.plot(x_fit, y_lin_fit, linestyle=':', label="linear fit(d=1), $R^2=%.2f$"%model_r2, c='blue', lw=3)  # lw는 선의 두께
plt.plot(x_fit, y_quad_fit, linestyle='-', label="quad fit(d=2), $R^2=%.2f$"%q_r2, c='red', lw=3)  # lw는 선의 두께
plt.plot(x_fit, y_cubic_fit, linestyle='--', label="cubic fit(d=3), $R^2=%.2f$"%c_r2, c='green', lw=3)  # lw는 선의 두께

plt.xlabel('하위계층 비율')
plt.ylabel('주택가격')
plt.legend()
plt.show()

# 주의 : 비선형 모델이 무조건 좋은 것은 아니다. 원본 데이터 재가공을 고민
# 모델 선택시 분석가의 경험이나 능력이 중요하다.



