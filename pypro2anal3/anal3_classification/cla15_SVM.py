# XOR(exclusive or) 문제는 선형으로 분류 불가. kernel trick을 사용하는 SVM으로 분류 가능. 차원 증가로 가능해진다.

x_data = [      # xor case
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0],
]

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

x_df = pd.DataFrame(x_data)
feature = np.array(x_df.iloc[:,0:2])
label = np.array(x_df.iloc[:,2])
print(feature)
print(label)

# 실습 1 : LogisticRegression
model = LogisticRegression()
model.fit(feature, label)
pred = model.predict(feature)
print('예측 -', pred)
print('실제 -', label)
print('정확도 : ', metrics.accuracy_score(label, pred))    # 정확도 :  0.75
print("-------"*20)

# 실습 2 : SVC
model2 = svm.SVC(C=1)       # C 인자는 과적합 방지를 위해 넣어준다.
# model2 = svm.LinearSVC(C=1)       # SVC에 비해 속도가 향상
model2.fit(feature, label)
pred2 = model2.predict(feature)
print('예측 -', pred2)
print('실제 -', label)
print('정확도 : ', metrics.accuracy_score(label, pred2))    # 정확도 :  0.75

