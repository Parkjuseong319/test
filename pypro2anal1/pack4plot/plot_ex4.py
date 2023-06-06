# 자전거 공유 시스템 분석용
#  : kaggle 사이트의 Bike Sharing in Washington D.C. Dataset를 편의상 조금 변경한 dataset을 사용함
#
# columns : 
#  'datetime', 
#  'season'(사계절:1,2,3,4), 
#  'holiday'(공휴일(1)과 평일(0)), 
#  'workingday'(근무일(1)과 비근무일(0)), 
#  'weather'(4종류:Clear(1), Mist(2), Snow or Rain(3), Heavy Rain(4)), 
#  'temp'(섭씨온도), 'atemp'(체감온도), 
#  'humidity'(습도), 'windspeed'(풍속), 
#  'casual'(비회원 대여량), 'registered'(회원 대여량), 
#  'count'(총대여량)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylabel
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

train = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/data/train.csv", parse_dates=['datetime'])
print(train.info())
print(train.shape)      # (10886, 12) 행, 열
print(train.columns)
print(train.head(2))

print(train.isnull().sum())     # null값 개수 확인
# null 값 확인 시각화
# import missingno as msno        # pip install missingno
# msno.matrix(train, figsize=(12, 5))
# plt.show()

# boxplot
fig, axes = plt.subplots(nrows=2, ncols=2)



fig.set_size_inches(12, 10)
sns.boxplot(data=train, y='count', orient='v', ax=axes[0][0])
sns.boxplot(data=train, y='count',x='season', orient='v', ax=axes[0][1])
sns.boxplot(data=train, y='count',x='holiday', orient='v', ax=axes[1][0])
sns.boxplot(data=train, y='count',x='workingday', orient='v', ax=axes[1][1])
axes[0][0].set(ylabel='count', title='대여량')
plt.show()





