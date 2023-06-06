# 다항분류 : LogisticRegression을 사용 - 활성화 함수는 softmax. sigmoid가 아닌 softmax로 출력한다.

import numpy as np
import matplotlib.pyplot as plt
"""
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
 
x = np.array([1.0,0.3,0.213])
y = softmax(x)
print(y)
print('가장 큰 인덱스는 ',np.argmax(y))
print(np.sum(y))
 
ratio = y
labels = y
 
plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression     # 다중 클래스(레이블)를 지원. 즉 출력결과가 다양하다.
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris = datasets.load_iris()
# print(iris.DESCR)
print(iris.keys())
# print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
print(iris.feature_names)       # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# x = iris.data
# print(x)        # metrix의 타입으로 되어있다.
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))  # 0.96286543 . 매우 강한 상관관계.
x = iris.data[:,[2,3]]  #  'petal length (cm)', 'petal width (cm)'만 가져다가 사용.
y = iris.target
print(x[:2], x.shape)  # [[1.4 0.2] [1.4 0.2]] (150, 2)   매트릭스
print(y[:2], y.shape)   # [0 0] (150,)        벡터

# 과적합 방지를 목적으로 train(모델 학습용) / test(모델 검정용)로 데이터를 분리 (7:3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     # (105, 2) (45, 2) (105,) (45,)

"""
print("--------"*10)
# 변수 데이터 크기가 너무 크거나 제각각일 경우(단위가 다를경우에도) 표준화, 정규화를 할 수 있다. (scaling). 대상은 독립변수(feature)    많이 권장한다.
print(x_train[:3])
sc = StandardScaler()       # 표준화 하는 객체
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])

# 스케일 결과 원상복구
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])

print("--------"*10)
"""
print('분류 모델 생성')
# from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier     
model = LGBMClassifier(boosting_type='gbdt', n_estimators=100, random_state=1, n_jobs=1)     # n_jobs는 처리할 cpu 개수를 얘기한다.
print(model)
model.fit(x_train, y_train)

# 분류 예측s
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총 개수:%d, 오류 수 :%d'%(len(y_test), (y_test != y_pred).sum()))  # 총 개수:45, 오류 수 :1
print('분류정확도 확인 방법 1 ')
print('%.5f'%accuracy_score(y_test, y_pred))    # 0.97778

print('분류정확도 확인 방법 2')
con_mat = pd.crosstab(y_test,y_pred,rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))    #0.97777777777777

print('분류정확도 확인 방법 3')
print('test : ', model.score(x_test, y_test))   #  0.9777777777777777
print('train : ', model.score(x_train, y_train))    # 0.971428571
# 1번, 2번과는 성격이 조금 다르다.
# 학습의 분류 정확도와 검정의 분류 정확도가 많이 차이나면 안된다. 학습 분류정확도가 너무 높다면 과적합(overfitting)상태이다.

# 최적의 학습모델이 생성되었다면 저장. 매번 학습할 필요 x 오히려 안좋다.
import pickle
pickle.dump(model, open('logicmodel.sav', 'wb'))    # 확장자는 기존에 있는 확장자명이 아닌 확장자로.

del model   # 저장했기 때문에 모델 삭제해도 상관없다.

print('새로운 값으로 분류 예측 "petal length (cm)", "petal width (cm)"')
print(x_test[:3])   # [[5.1 2.4] [4.  1. ] [1.4 0.2]]
new_data = np.array([[5.1, 2.4], [4., 1.], [1.4, 0.2], [1. , 1.], [7. ,7.]])
# 참고 : 표준화를 한 경우라면 sc.fit(new_data); new_data = sc.tranform(new_data)

mymodel = pickle.load(open('logicmodel.sav', 'rb'))
new_pred = mymodel.predict(new_data)
print('예측결과 : ', new_pred)      # [2 1 0 0 2]
print('softmax 결과 : ', mymodel.predict_proba(new_data))

# 시각화 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager, rc

plt.rc('font', family='malgun gothic')      
plt.rcParams['axes.unicode_minus']= False

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=mymodel, test_idx=range(105, 150), title='scikit-learn제공')     

