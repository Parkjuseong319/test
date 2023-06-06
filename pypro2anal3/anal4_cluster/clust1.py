# 군집분석
# - 개인 또는 여러 개체를 유사한 속성을 지닌 대상들 끼리 그룹핑 하는 탐색적 다변량 분석기법이다.
# - 거리값(Distance Measure)을 이용해 가까운 거리에 있는 것들끼리 묶어 분류한다.
# - 계층적 군집분석과 비계층적 군집으로 분류할 수 있다.
# 1) 계층적 군집분석 :
# 개별 대상 간의 거리에 의하여 가장 가까이 있는 대상들로 부터 시작하여 결합해 감으로써 나무모양의
# 계층적 구조를 형성해 나가는 방법으로 이 과정에서 군집의 수가 감소한다. 계층적 군집분석은 군집이
# 형성되는 과정을 정확하게 파악할 수 있다는 장점이 있으나 자료의 크기가 크면 분석하기 어렵다는 단점이
# 있다.
# 방법 : 단일결합법, 완전결합법, 평균결합법, 중심결합기준법, Ward법

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

np.random.seed(123)

var = ['x', 'y']
labels = ['점0', '점1', '점2', '점3', '점4']
x = np.random.random_sample([5,2]) * 10 
print(x)
df = pd.DataFrame(x, columns=var, index=labels)
print(df)

# plt.scatter(x[:, 0], x[:,1], s=50, c='blue', marker='o')
# plt.grid(True)
# plt.show()

from scipy.spatial.distance import pdist, squareform    # pdist: 유클리디안 거리 계산법으로 거리를 재준다.

dist_vec = pdist(df, metric='euclidean')
print(dist_vec)

row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)
print(row_dist)

from scipy.cluster.hierarchy import linkage
# linkage : 분할적X, 병합적 O. 다양한 method 기법으로 응집형(병합)형 계층적 클러스터를 수행
row_clusters = linkage(dist_vec, method='ward')     # 'ward', 'single', 'average', ...
# print(row_clusters)
df = pd.DataFrame(row_clusters, columns=['군집1', '군집2', '거리', '멤버수'])
print(df)

from scipy.cluster.hierarchy import dendrogram
row_dend = dendrogram(row_clusters, labels=labels)
plt.ylabel('유클리드 거리')
plt.tight_layout()
plt.show()




