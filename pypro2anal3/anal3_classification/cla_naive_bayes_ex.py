"""
독버섯(poisonous)인지 식용버섯(edible)인지 분류
https://www.kaggle.com/datasets/uciml/mushroom-classification
feature는 중요변수를 찾아 선택, label:class
참고 : from xgboost import plot_importance


데이터 변수 설명 : 총 23개 변수가 사용됨.

"""
from xgboost import plot_importance
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.decomposition import PCA

data = pd.read_csv('../testdata/mushrooms.csv')
pd.set_option('display.max_columns', 30)
print(data.head(3))

la = data.apply(LabelEncoder().fit_transform)       
# LabelEncoder().fit_transform() 는 2차원은 인코딩 불가. 
# DataFrame이 아니라 Series 형태로 넣거나 apply함수를 이용해서 칼럼 여러개를 적용시킬 수 있다.
feature = la.drop(columns=['class'])
label = data['class'].apply(lambda x:0 if x=='e' else 1)
print(feature[:3])
print(label[:3], label.shape)

x_train, x_test, y_train, y_test = train_test_split(feature,label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) 

# 모델 생성
model = xgb.XGBClassifier(booster="gbtree", max_depth=6, n_estimator=500).fit(x_train, y_train)
pred = model.predict(x_test)
print('예측 -', pred[:15])
print('실제 -', y_test[:15].values)
print('분류정확도 : ',metrics.accuracy_score(y_test, pred))

# 시각화 
fig, ax = plt.subplots(figsize = (10,12))
xgb.plot_importance(model, ax=ax)
plt.show()


# model = GaussianNB().fit(x_traijn,y)
# print(model)
# pred = model.predict(x)
# print(pred)
# print('분류정확도 : ',metrics.accuracy_score(y, pred))
