# 과적합 방지 방안 : train_test split, cross validation(K-Fold)
# 데이터에 따라 차이가 있으나 최적화된 학습 모델로 새로운 데이터에 대한 일반화 처리(Classification, Regression)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
train_data = iris.data
train_label = iris.target

print(train_data[:2], train_data.shape)
print(train_label[:2], train_label.shape)

# 분류 모델 생성
dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data)
print('예측값 : ', pred)
print('실제값 : ', train_label)
print('분류정확도 : ', accuracy_score(train_label, pred))        #  1.0 <<< overfitting 의심. 포용성이 떨어진다고 보면 된다.

print('과적합 방지 처리 1 : 데이터의 양이 많으나 과적합 발생한 경우 - train_test split 권장')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=121)
dt_clf.fit(x_train,y_train)
pred2 = dt_clf.predict(x_test)  # test dataset으로 학습모델 검증
print('예측값 : ', pred2[:10])
print('실제값 : ', y_test[:10])
print('분류정확도(split) : ', accuracy_score(y_test, pred2))    # 포용성이 있는 모델이 만들어짐

print('\n과적합 방지 처리 2 : train_test split 후에도 과적합 의심이 된다면 cross validation(교차검증) 권장') # 처음부터 교차검증 가능
# train data를 train / validation data로 분리 후 학습
# 교차검증 중 가장 보편적인 방법은 K-Fold
# K-Fold : K개의 data fold set을 만들어, 학습도중에 K번 만큼 학습과 평가를 병행함
# https://nonmeyet.tistory.com/entry/KFold-Cross-Validation%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D-%EC%A0%95%EC%9D%98-%EB%B0%8F-%EC%84%A4%EB%AA%85

from sklearn.model_selection import KFold
import numpy as np

features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=121)
kfold = KFold(n_splits=5)
cv_acc = []
print('iris shape : ', features.shape)      # (150, 4)
# 전체 행 수 : 150개. 학습데이터 4 / 5(120개), 검증데이터 : 1 / 5(30개)로 분할해서 학습할 예정

n_iter = 0
for train_index, test_index in kfold.split(features):
    # print('n_iter : ', n_iter)
    # print('train_index : ', train_index, ' ', len(train_index))
    # print('test_index : ', test_index, ' ', len(test_index))
    # n_iter += 1
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    # 학습 및 예측
    dt_clf.fit(xtrain, ytrain)
    pred = dt_clf.predict(xtest)
    n_iter += 1
    # 반복할 때 마다 분류 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]          # 여기서 test는 사실 validation이다.
    # print('반복수:{0}, 교차검증 정확도:{1}, 학습데이터 크기:{2}, 검정데이터 크기:{3}'.format(n_iter, acc, train_size, test_size))
    # print('반복수:{0}, test set index:{1}'.format(n_iter, test_index))
    
    cv_acc.append(acc)
    
print('평균 검증 정확도 : ', np.mean(cv_acc))  # 5회 교차검증 결과 분류 정확도 : 0.9132    


print('\n과적합 방지 처리 3 : 교차검증을 함수로 지원')
from sklearn.model_selection import cross_val_score
data = iris.data
label = iris.target
score = cross_val_score(dt_clf, X=data, y=label, scoring='accuracy', cv=5)  # cv는 k-fold에서 k값과 같음.
print('교차 검증별 정확도 : ', np.round(score, 3))
print('평균 검증 정확도 : ', np.round(np.mean(score), 3))      # 0.96

print('\n과적합 방지 처리 4 : 불균형한 분포(편향, 왜곡)를 가진 레이블 데이터 집합을 위한 교차검증 class : StratifiedKFold')
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5)
cv_acc = []
n_iter = 0

for train_index, test_index in skfold.split(features, label):
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    # 학습 및 예측
    dt_clf.fit(xtrain, ytrain)
    pred = dt_clf.predict(xtest)
    n_iter += 1
    # 반복할 때 마다 분류 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]          # 여기서 test는 사실 validation이다.
    print('반복수:{0}, 교차검증 정확도:{1}, 학습데이터 크기:{2}, 검정데이터 크기:{3}'.format(n_iter, acc, train_size, test_size))
    print('반복수:{0}, test set index:{1}'.format(n_iter, test_index))
    
    cv_acc.append(acc)
print('교차 검증별 정확도 : ', np.round(cv_acc, 3))
print('평균 검증 정확도 : ', np.round(np.mean(cv_acc), 3))      # 0.96
