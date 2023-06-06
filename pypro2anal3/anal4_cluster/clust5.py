# 밀도기반 clustering : 비선형 데이터의 군집분류 가능
# DBSCAN 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

x, _ = make_moons(n_samples= 200, noise = 0.05, random_state=0)
print(x)

# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

# KMeans로 군집화
km = KMeans(n_clusters = 2, random_state=0)
pred1 = km.fit_predict(x)
print('군집 id : ', pred1[:10])

def plotResult(x, pr):
    plt.scatter(x[pr==0, 0], x[pr==0, 1], c='blue', marker='o', s=40, label='clus1')
    plt.scatter(x[pr==1, 0], x[pr==1, 1], c='red', marker='s', s=40, label='clus1')
    plt.legend()
    plt.show()

# plotResult(x, pred1)
# KMeans는 원하는 군집을 형성하지 못함

print()
dm = DBSCAN(eps = 0.2, min_samples=5, metric='euclidean')
pred2 = dm.fit_predict(x)
print(pred2)



