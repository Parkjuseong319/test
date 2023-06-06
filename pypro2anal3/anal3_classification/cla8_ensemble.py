# Ensemble Learning : 개별적으로 동작하는 모델들을 모아 종합적으로 의사결정하기 위함
# Voting, Bagging, Boosting
# Wisconsin Diagnostic Breast Cancer (WDBC) : 위스콘신 유방암 진단 데이터
# Logistic Regression, DecisionTree, KNN를 사용
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cancerdata = load_breast_cancer()
data_df = pd.DataFrame(cancerdata.data, columns = cancerdata.feature_names)
print(data_df.head(3), data_df.shape)       # (569, 30)

x_train, x_test, y_train, y_test = train_test_split(cancerdata.data, cancerdata.target, test_size=0.2, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape )    # (455, 30) (114, 30) (455,) (114,)

# VotingClassifier에 넣을 개별 모델 객체를 생성.
logi_regression = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors = 3)
dec_model = DecisionTreeClassifier()

classifiers = [logi_regression,knn,dec_model]

# 개별 모델이 학습 및 예측 평가를 먼저 확인해보자
for classifier in classifiers:     
    classifier.fit(x_train, y_train) 
    pred = classifier.predict(x_test)
    class_name = classifier.__class__.__name__
    print('{0} 정확도 : {1:.3f}'.format(class_name, accuracy_score(y_test,pred)))
# LogisticRegression 정확도 : 0.939
# KNeighborsClassifier 정확도 : 0.895
# DecisionTreeClassifier 정확도 : 0.939

# VotingClassifier로 확인
voting_model = VotingClassifier(estimators=[('LR',logi_regression),('KNN',knn), ('Decision',dec_model)], voting='soft')

voting_model.fit(x_train, y_train)
vpred = voting_model.predict(x_test)
print('Voting 분류기의 정확도 : {:.3f}'.format(accuracy_score(y_test,vpred)))  # hard : 0.939, soft : 0.947






