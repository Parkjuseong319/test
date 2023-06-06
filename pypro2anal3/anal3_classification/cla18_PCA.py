# 특성공학의 방법 중 하나로 
# 주성분 분석(Principal component analysis; PCA)은 고차원의 데이터를 저차원의 데이터로 변환시키는 기법을 말한다.
# 이 때 서로 연관 가능성이 있는 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간(주성분)의 표본으로 변환하기 위해 직교 변환을 사용한다. 
# 데이터를 한개의 축으로 사상시켰을 때 그 분산이 가장 커지는 축을 첫 번째 주성분, 두 번째로 커지는 축을 두 번째 주성분으로
# 놓이도록 새로운 좌표계로 데이터를 선형 변환한다.
# PCA는 전체 분석 과정 중 초기단계에서 발생한다.

# iris dataset으로 차원축소
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')

iris = load_iris()
print(iris.data[:3])
print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# 'sepal length', 'sepal width' 두 개의 변수의 패턴이 유사한지 확인 후 PCA 적용

n = 10
x = iris.data[:n, :2]
print('차원 축소 전 x : ',x[:2], x.shape)  # (150, 2)
print(x.T)      # [[5.1 4.9 4.7 4.6 5.  5.4 4.6 5. ...] [3.5 3.  3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...]]

# plt.plot(x.T, 'o:')
# plt.grid(True)
# plt.title('iris 크기 특성')
# plt.xlabel('특성 종류')
# plt.ylabel('특성 값')
# plt.legend(["표본 {}".format(i + 1) for i in range(n)])
# plt.show()

df = pd.DataFrame(x)
# ax = sns.scatterplot(x=df[0],y= df[1], data = df, marker='s', s=100, color=['b'])  # s는 scale(크기).
# for i in range(n):
#     ax.text(x[i,0] -0.05, x[i,1] + 0.02, '표본 {}'.format(i + 1))
# plt.xlabel('꽃받침길이')
# plt.ylabel('꽃받침넓이')
# plt.title('iris 크기 특성(2차원)')
# plt.axis('equal')
# plt.show()
# 위 그림으로 길이와 너비가 정비례함을 알 수 있다. 패턴이 유사하다. 이것은 차원축소의 근거가 될 수 있다.

# PCA 기능 사용
pca1 = PCA(n_components = 1)    # 변환할 차원의 수
x_low = pca1.fit_transform(x)   # 비지도 학습으로 y를 지정하지 않음
print('x_low : ', x_low, ' ', x_low.shape)  # (10, 1). 원래는 (10, 2)였다. 차원 축소 됨을 알 수 있음.

x2 = pca1.inverse_transform(x_low)
print('차원 복귀값 : ', x2, ' ', x2.shape)       # (10, 2)
print('최초값 ',x[0, :])       # [5.1 3.5]
print('차원축소값 ',x_low[0])    # [0.30270263]
print('차원축소에서 원복한 값',x2[0, :])  # [5.06676112 3.53108532]    주성분 분석으로 인해 버려지는 값들이 있어서 최초값의 근사값으로 반환된다.

# 차원 축소된 자료 포함 시각화
# ax = sns.scatterplot(x=df[0],y= df[1], data = df, marker='s', s=100, color=['b'])  # s는 scale(크기).
# for i in range(n):
#     ax.text(x[i,0] -0.05, x[i,1] + 0.02, '표본 {}'.format(i + 1))
#     plt.plot([x[i, 0], x2[i, 0]], [x[i, 1], x2[i, 1]], 'k--' )
# plt.plot(x2[:, 0], x2[:, 1], 'o-', color='c', markersize=10)        # 축소된 값 projection 된 값.
# plt.plot(x[:, 0].mean(), x[:, 1].mean(), marker='D', color='r', markersize=10)  # 실제값 평균
# plt.xlabel('꽃받침길이')
# plt.ylabel('꽃받침넓이')
# plt.title('iris 크기 특성(2차원)')
# plt.axis('equal')
# plt.show()

print('iris dataset 차원축소')
x = iris.data
print(x[:2])
pca2 = PCA(n_components = 2)        # 4개의 열을 2개로 차원축소
x_low2 = pca2.fit_transform(x)
print('x_low2 : ', x_low2, ' ', x_low2.shape)
print(pca2.explained_variance_ratio_)   
# 전체 변동성에서 각 PCA component 별로 차지하는 변동성 비율을 출력
# [0.92461872 0.05306648] 제1주성분, 제2주성분,...  1 - 두 개 값의 합은 버려진 주성분들의 비율이다. 

x4 = pca2.inverse_transform(x_low2)
print('최초값 ',x[0, :])           # [5.1 3.5 1.4 0.2] 
print('차원축소값 ',x_low2[0])    # [-2.68412563  0.31939725]
print('차원축소에서 원복한 값',x4[0, :]) # [5.08303897 3.51741393 1.40321372 0.21353169]
iris1 = pd.DataFrame(x, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
iris2 = pd.DataFrame(x_low2, columns=['var1', 'var2'])
print(iris1.head(3))
print(iris2.head(3))


