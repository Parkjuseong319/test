# K-최근접 이웃(K-NN, K-Nearest Neighbor) 알고리즘은 가장 간단한 머신러닝 알고리즘으로, 분류(Classification) 알고리즘이다.
# 비슷한 특성을 가진 데이터는 비슷한 범주에 속하는 경향이 있다는 가정하에 사용한다.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import mglearn      # pip install mglearn
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# 가장 간단한 k-NN 알고리즘은 가장 가까운 훈련 데이터 포인트 하나를 최근접 이웃으로 찾아 예측에 사용한다. 
# 단순히 이 훈련 데이터 포인트의 출력이 예측이 된다.

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)
print(X_train, ' ', X_train.shape)  # [[ 8.92229526 -0.63993225] ...   (19, 2)
print(X_test, ' ', X_test.shape)    #  (7, 2)
print(y_train)  # [0 0 1 1 0 1 0 1 1 1 0 1 0 0 0 1 0 1 0]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print("test 예측: {}".format(model.predict(X_test)))
print("test 정확도: {:.2f}".format(model.score(X_test, y_test)))
print("train 정확도: {:.2f}".format(model.score(X_train, y_train)))

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for n_neighbors, ax in zip([1, 3, 9], axes):
    model2 = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(model2, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
    axes[0].legend(loc=1)

plt.show()

# -------------------------
# Classification
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=5)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()
# -------------------------
