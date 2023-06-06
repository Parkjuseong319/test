# iris dataset으로 지도학습(K-NN) / 비지도학습(K-Means)

from sklearn.datasets import load_iris
import pandas as pd

iris_data = load_iris()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y =  train_test_split(iris_data['data'], iris_data['target'],test_size=0.3, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)     # (112, 4) (38, 4) (112,) (38,)

print('지도학습(K-NN) ----------------')
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
knnModel.fit(train_x, train_y)      # feature, label
print('모델 성능')
from sklearn import metrics
pred = knnModel.predict(test_x)
print('acc : ', metrics.accuracy_score(test_y, pred))

# 새로운 값으로 분류 예측
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
print('new_input : ', knnModel.predict(new_input))
print('new_input : ', knnModel.predict_proba(new_input))
dist, index = knnModel.kneighbors(new_input)
print(dist, index)

print('\n비지도학습(KMeans) ---------------')
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters=3, init = 'k-means++', random_state=42)
kmeansModel.fit(train_x)        # feature만 있다. label은 안주는 것이 비지도학습
print(kmeansModel.labels_)
print('0군집 : ', train_y[kmeansModel.labels_ == 0])
print('1군집 : ', train_y[kmeansModel.labels_ == 1])
print('2군집 : ', train_y[kmeansModel.labels_ == 2])

new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
clu_pred = kmeansModel.predict(new_input)
print(clu_pred)

print('군집 모델 성능 측정')
predict_cluster = kmeansModel.predict(test_x)
print(predict_cluster)

np_arr = np.array(predict_cluster)
np_arr[np_arr == 0],np_arr[np_arr == 1],np_arr[np_arr == 2] = 3,4,5 
np_arr[np_arr == 3] = 1     # 군집3(군집0)을 1(versicolour)로 변경
np_arr[np_arr == 4] = 0     # 군집4(군집1)을 0(setosa)로 변경
np_arr[np_arr == 5] = 2     # 군집5(군집2)을 2(verginica)로 변경
print(np_arr)

predict_label = np_arr.tolist()
print(predict_label)
print('test acc : {:.2f}'.format(np.mean(predict_label ==test_y)))
