# iris dataset으로 시각화 연습
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
# 참고 : ipython 기반의 jupyter를 사용할 경우 %matplotlib inline 하면 plt.show() 생략

iris_data = pd.read_csv('../testdata/iris.csv')
print(iris_data.head(3))
print(iris_data.info())

print(iris_data['Species'].unique())
print(set(iris_data['Species']))

# 1, 3 열을 이용해 산점도
plt.scatter(iris_data["Sepal.Width"],iris_data["Petal.Width"])
plt.show()

# pandas의 시각화
iris_col = iris_data.loc[:, ['Sepal.Width','Petal.Width']]
from pandas.plotting import scatter_matrix
scatter_matrix(iris_col)
plt.show()

# Seaborn : Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
import seaborn as sns

sns.pairplot(iris_data, hue='Species', height=2)
plt.show()

sns.kdeplot(iris_data['Sepal.Width'].values)    # 밀도 곡선
plt.show()
