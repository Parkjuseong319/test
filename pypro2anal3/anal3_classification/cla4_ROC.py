# 분류 모델 성능 평가시 반드시 기술해야할 사항
# 모델의 정확도(Accuracy), 정밀도(Precision), 민감도(재현률, recall, sensitivity)
# ROC curve 를 사용 - FPR이 변할 때 TPR이 어떻게 변하는지 알려주는 곡선(0~1사이의 값을 가짐)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
# 표본 100개, 독립변수 2개 중 다른 독립변수의 선형조합으로 나타나는 성분 수
print(x[:3])
print(y[:3])

import matplotlib.pyplot as plt     # 산포도 or boxplot 사용하는게 좋다.

# plt.scatter(x[:, 0], x[:, 1])
# plt.scatter(x[:, 0], y)
# plt.show()

# ... test / train 생략 ...

model = LogisticRegression().fit(x,y)
y_hat = model.predict(x)
print('예측 값 : ', y_hat[:10])
print('실제 값 : ', y[:10])

f_value = model.decision_function(x)   
# decision_function() : 결정(판별) 함수 - 결정/판별/불확실성 추정 합수. ROC curve의 판별 경계선 설정을 위한 sample data 제공
print('f value : ', f_value)
print()
df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=['f', 'y_hat', 'y'])
print(df)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_hat))

"""
       모델 예측
        P   N
실제 T  TP  FN
실제 F  FP  TN
"""
acc = (44 + 44) / 100       # TP + TN / 전체 수
recall = 44 / 48            # TP / (TP + FN)
precision = 44 / 52         # TP / (TP + FP)
specificity = 44 / (44 + 8) # TN / (FP + TN)    TPR
# 민감도는 양성률이라고도 불리며, 위양성률은 1 – 특이도로 계산
fallout = 8 / (44 + 8)       # FP / (FP + TN)    FPR
print('acc(정확도)',  acc)
print('recall(재현률)',  recall)
print('precision(정밀도)',  precision)
print('specificity(특이도)',  specificity)
print('fallout(위양성률)',  fallout)
print('fallout(위양성률)',  1 - specificity)
# TPR은 1에 가까울수록 좋고, FPR은 0에 가까울 수록 좋다.

print()
from sklearn import metrics
ac_sco = metrics.accuracy_score(y, y_hat)
print('ac_sco : ', ac_sco)      # 정확도 0.88
cl_rep = metrics.classification_report(y, y_hat)
print('cl_rep : ', cl_rep)
print()
fpr, tpr, threshold = metrics.roc_curve(y, model.decision_function(x))      # threshold는 분류 결정 임계치이다.
print('fpr : ', fpr)
print('tpr : ', tpr)
# print('threshold : ', threshold)

# ROC curve
plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='classifier line(AUC:0.5)')
plt.plot([fallout], [recall], 'ro', ms=20)      # 위양성률과 재현율 값
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

# AUC(Area Under the Curve) : ROC curve의 밑면적 - 1에 근사하면 좋음
print('AUC :', metrics.auc(fpr, tpr))      # 0.954727564






