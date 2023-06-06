# kaggle.com이 제공하는 'glass datasets'
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
# RI    Na    Mg    Al    Si    K    Ca    Ba    Fe    Type
#                           ...
# glass.csv 파일을 읽어 분류 작업을 수행하시오.

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../testdata/glass.csv")
print(data.head(3), data.shape)
# print(data['Type'].values)

x = data[['RI', 'Na','Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = LabelEncoder().fit_transform(data['Type'])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (171, 9) (43, 9) (171,) (43,)

# 모델 생성
model = xgb.XGBClassifier(booster="gbtree", max_depth=6, n_estimator=500).fit(x_train, y_train)
pred = model.predict(x_test)
print('예측 -', pred[:15])
print('실제 -', y_test[:15])

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print('분류 정확도 : ',acc)

# 시각화 
fig, ax = plt.subplots(figsize = (10,12))
xgb.plot_importance(model, ax=ax)
plt.show()
