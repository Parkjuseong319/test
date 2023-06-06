# Boosting 이란?
# 여러 개의 약한 Decision Tree를 조합해서 사용하는 Ensemble 기법 중 하나이다.
# 즉, 약한 예측 모형들의 학습 에러에 가중치를 두고, 순차적으로 다음 학습 모델에 반영하여
# 강한 예측모형을 만드는 것이다.

# XGBoost 란?
# XGBoost는 Extreme Gradient Boosting의 약자이다.
# Boosting 기법을 이용하여 구현한 알고리즘은 Gradient Boost 가 대표적인데
# 이 알고리즘을 병렬 학습이 지원되도록 구현한 라이브러리가 XGBoost 이다.
# Regression, Classification 문제를 모두 지원하며, 성능과 자원 효율이 좋아서, 인기 있게 사용되는 알고리즘이다.

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import plot_importance

dataset = load_breast_cancer()
print(dataset.keys())
x_feature = dataset.data
y_label = dataset.target
# print(dataset.feature_names)
print(x_feature)
print(y_label)
cancer_df = pd.DataFrame(data=x_feature, columns=dataset.feature_names)
# pd.set_option('display.max_columns', None)          # 콘솔창에서 생략되는 열들을 보이게 하는 함수
print(cancer_df.head(2), cancer_df.shape)   # (569, 30)
print(dataset.target_names)     # ['malignant' 'benign']
print(np.sum(y_label == 0))     # malignant(양성) : 212
print(np.sum(y_label == 1))     # benign(음성) : 357

x_train, x_test, y_train, y_test = train_test_split(x_feature, y_label, test_size=0.2, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (455, 30) (114, 30) (455,) (114,)

# > pip install xgboost
model = xgb.XGBClassifier(booster="gbtree", max_depth=6, n_estimator=500).fit(x_train,y_train)     # Bagging 알고리즘 사용

# > pip install lightgbm
from lightgbm import LGBMClassifier     # Boosting 알고리즘 사용
# model = LGBMClassifier(boosting_type='gbdt', n_estimators=500, max_depth=6).fit(x_train,y_train)
print(model)

pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10])

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print('분류 정확도 : ',acc)

# 시각화는 XGBClassifier만 지원된다.  LGBMClassifier 사용 불가
fig, ax = plt.subplots(figsize = (10,12))
plot_importance(model, ax=ax)        
plt.show()

