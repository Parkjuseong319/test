# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
# dataset은 winequality-red.csv 
# https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv

# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../testdata/winequality_red.csv')
# print(df.head(4), df.shape)

x = df.iloc[:,0:11]
y = df['quality']
print(x.head(3))
print(min(y), max(y))
print(df.corr())

# train / test split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
print(train_x[:2])
print(train_y[:2])

# 표준화 
sc = StandardScaler()       # 표준화 하는 객체
sc.fit(train_x); sc.fit(test_x)
x_train = sc.transform(train_x)
x_test = sc.transform(test_x)
print(x_train[:3])

# 모델 생성
model = RandomForestClassifier(criterion='entropy', n_estimators=1000).fit(train_x, train_y)
pred = model.predict(test_x)
print("예측값 : ", pred[:10])
print("실제값 : ", np.array(test_y[:10]))

# 분류 정확도
print('acc : ', sum(test_y == pred) / len(test_y))
print('acc2 : ', accuracy_score(test_y, pred))

# 교차 검증
from sklearn.model_selection import cross_val_score
vali = cross_val_score(model, train_x, train_y, cv=5)
print(np.around(np.mean(vali),3))   # 약 0.671

# 중요변수 확인하기
print('특성(변수) 중요도 : {}'.format(model.feature_importances_))
# alcohol이 특성 중 가장 중요하다.

# 중요 변수 시각화
import matplotlib.pyplot as plt

n_features = x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('attr importance')
plt.ylabel('attr')
plt.yticks(np.arange(n_features), x.columns)
plt.show()


print(np.mean(df[df['quality']==3]['alcohol']))
print(np.mean(df[df['quality']==8]['alcohol']))

# what does quality mean from the number 3,4,5,6,7,8. 
# Are these numbers the best quality to low quality?
# 알코올 수치에 따른 품질을 얘기한다.
# quality가 3일때의 alcohol 평균은 9.995
# quality가 8일때의 alcohol 평균은 12.094
# quality가 8일때 best quality, quality가 3일때 low quality.

# 임의의 와인 정보를 받아 quality 예측하기
# indata = pd.DataFrame({'alcohol':float(input('와인의 알코올 수치를 입력하세요'))})
# re_pred = model.predict(indata)
# print('와인 퀄리티는 {}점 입니다.'.format(re_pred))


