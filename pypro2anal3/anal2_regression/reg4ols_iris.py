# iris dataset으로 선형 회귀분석
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import seaborn as sns

iris = sns.load_dataset("iris")
print(iris.head(3))

print(iris.corr())

# 연습1 : 상관관계가 약한 변수를 사용
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print('요약결과 1 : ', result1.summary())
print('R squared : ', result1.rsquared)
print('p-value : ', result1.pvalues[1])
# 연습1 같은 경우에는 의미없는 모델이다. 두 변수 간의 인과관계는 없다.

plt.scatter(iris.sepal_width, iris.sepal_length)
plt.plot(iris.sepal_width, result1.predict(), color='r')
plt.show()

# 연습2 : 상관관계가 강한 변수를 사용
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
print('요약결과 2 : ', result2.summary())
print('R squared : ', result2.rsquared)
print('p-value : ', result2.pvalues[1])
# 연습2 같은 경우에는 의미있는 모델이다. 두 변수 간의 인과관계가 있다.

plt.scatter(iris.petal_length, iris.sepal_length)
plt.plot(iris.petal_length, result2.predict(), color='b')
plt.show()

print('-----------------다중선형회귀(독립변수 복수)-----------------')
# result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width + sepal_width', data=iris).fit()
# result3 = smf.ols(formula='sepal_length ~ .', data=iris).fit()    # R 처럼 사용불가
column_select = '+'.join(iris.columns.difference(['sepal_length','species']))   # 여러개 칼럼 독립변수로 둘때 이 방식으로 넣어주기도 한다.
print(column_select)        # petal_length+petal_width+sepal_width
result3 = smf.ols(formula='sepal_length ~' + column_select, data=iris).fit()

print('요약결과 3 : ', result3.summary())
print('R squared : ', result3.rsquared)
print('p-value : ', result3.pvalues[1])



