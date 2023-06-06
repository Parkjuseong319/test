# [GaussanNB 문제] 
# 독버섯(poisonous)인지 식용버섯(edible)인지 분류
# feature는 중요변수를 찾아 선택, label:class
# 참고 : from xgboost import plot_importance

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('../testdata/mushrooms.csv')

x_feature = dataset[dataset.columns.difference(['class'])].copy()
for column in x_feature.columns:
    x_feature[column] = LabelEncoder().fit_transform(x_feature[column])

y_label = LabelEncoder().fit_transform(dataset['class'])
    
# print(x_feature) 
# print(y_label) #식용 0 독버섯 1

x_train, x_test, y_train, y_test = train_test_split(x_feature,y_label,test_size=0.3,random_state=1)

model = xgb.XGBClassifier(booster='gbtree',max_depth=6,n_estimators=500).fit(x_train,y_train)
pred=model.predict(x_test)

from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('XGBClassifier 분류 정확도 :',acc) #0.1

#plot_importance 시각화. 
fig,ax = plt.subplots(figsize=(10,12))
plot_importance(model,ax=ax)
plt.show() #중요변수상위4개 (spore-print-color,odor,gill-size,cap-color)

#===나이브베이즈===

from sklearn.naive_bayes import GaussianNB

importance_x = x_feature[['spore-print-color','odor','gill-size','cap-color']]
model_nb = GaussianNB().fit(importance_x,y_label)
pred = model_nb.predict(importance_x)

print('나이브베이즈 분류 정확도 : ',metrics.accuracy_score(y_label,pred))
new_data = pd.DataFrame({"spore-print-color":[2],"odor":[6],"gill-size":[1],"cap-color":[4]})
print('예측:', '독버섯' if model_nb.predict(new_data) == 1 else '식용버섯')


