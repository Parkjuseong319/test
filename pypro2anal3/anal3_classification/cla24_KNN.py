# K-NN 모델에서 최적의 k값은?
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,stratify=cancer.target, random_state=55)
# stratify 인자는 쏠림현상을 방지시켜준다. 성능향상에 효과적 
train_accuracy = []
test_accuracy = []

neighbors_k = range(1, 11, 2)      # k값으로 짝수는 비권장

for n_neigh in neighbors_k:
    clf = KNeighborsClassifier(n_neighbors=n_neigh, weights='distance')
    clf.fit(x_train, y_train)
    train_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))
    
import numpy as np
print('train 평균 정확도 : ', np.mean(train_accuracy))
print('test 평균 정확도 : ', np.mean(test_accuracy))

plt.plot(neighbors_k, train_accuracy, label="훈련 정확도")
plt.plot(neighbors_k, test_accuracy, label="검정 정확도")
plt.xlabel('k값')
plt.ylabel('분류 정확도')
plt.legend()
plt.show()



