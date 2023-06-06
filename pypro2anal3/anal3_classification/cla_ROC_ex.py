# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
# 얘를 사용해도 됨   'testdata/advertisement.csv' 
# 참여 칼럼 : 
#   Daily Time Spent on Site : 사이트 이용 시간 (분)
#   Age : 나이,
#   Area Income : 지역 소독,
#   Daily Internet Usage:일별 인터넷 사용량(분),
#   Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간의 단위가 큰 경우 표준화 작업을 시도한다.
# ROC 커브와 AUC 출력
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

df = pd.read_csv('../testdata/advertisement.csv')
print(df.head(4))
# print(df.info())

x = df.loc[:,['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = df['Clicked on Ad']

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 표준화 
print(x_train[:3])
sc = StandardScaler()       # 표준화 하는 객체
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])

# 모델 생성
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0).fit(x_train, y_train)
y_pred = model.predict(x_test)
print('예측값 : ', y_pred[:5])
print('실제값 : ', y_test[:5])

# confusion table
con_mat = confusion_matrix(y_test, y_pred)
# print(con_mat)
"""
[[159   3]
 [  3 135]]
 """

acc = (con_mat[0][0] + con_mat[1][1]) / con_mat.sum()
recall = con_mat[0][0] / (con_mat[0][0] + con_mat[0][1])      # TP / (TP + FN)
precision = con_mat[0][0] / (con_mat[0][0] + con_mat[1][0])          # TP / (TP + FP)
specificity = con_mat[1][1] / (con_mat[1][0] + con_mat[1][1])  # TN / (FP + TN)    TPR
fallout = con_mat[1][0] / (con_mat[1][0] + con_mat[1][1])  
# print('acc(정확도)',  acc)
# print('recall(재현률)',  recall)
# print('precision(정밀도)',  precision)
# print('specificity(특이도)',  specificity)
# print('fallout(위양성률)',  fallout)
# print('fallout(위양성률)',  1 - specificity)

# ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test, model.decision_function(x_test))

plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='classifier line(AUC:0.5)')
plt.plot([fallout], [recall], 'ro', ms=20)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

print('AUC :', metrics.auc(fpr, tpr))       # AUC : 0.99838969

