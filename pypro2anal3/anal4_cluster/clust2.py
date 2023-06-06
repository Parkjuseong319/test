# iris dataset으로 계층적 군집 분석
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform 
plt.rc('font', family='malgun gothic')

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.head(3))

dist_vec = pdist(iris_df.loc[:, ['sepal length (cm)', 'sepal width (cm)']], metric='euclidean')
print(dist_vec)

row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist)

row_clusters = linkage(dist_vec, method='complete')     # complete : 완전 연결법
# print(row_clusters)
df = pd.DataFrame(row_clusters, columns=['군집1', '군집2', '거리', '멤버수'])
print(df)

from scipy.cluster.hierarchy import dendrogram
row_dend = dendrogram(row_clusters)
plt.ylabel('유클리드 거리')
plt.tight_layout()
plt.show()

print()
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
x = iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']]
labels = ac.fit_predict(x)
print('cluster result : ', labels)

plt.hist(labels)
plt.grid(True)
plt.show()




