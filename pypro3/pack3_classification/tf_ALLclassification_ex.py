# testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.
# * 변수 종류 *
# satisfaction_level : 직무 만족도
# last_eval‎uation : 마지막 평가점수
# number_project : 진행 프로젝트 수
# average_monthly_hours : 월평균 근무시간
# time_spend_company : 근속년수
# work_accident : 사건사고 여부(0: 없음, 1: 있음)
# left : 이직 여부(0: 잔류, 1: 이직)
# promotion_last_5years: 최근 5년간 승진여부(0: 승진 x, 1: 승진)
# sales : 부서

# salary : 임금 수준 (low, medium, high)

# 조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
# Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble._forest import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('../testdata/HR_comma_sep.csv')
print(data.head(10))
print(data.info())
# print(set(data['sales']))       # {'technical', 'management', 'support', 'sales', 'accounting', 'IT', 'product_mng', 'hr', 'RandD', 'marketing'}
# print(set(data['salary']))      # {'low', 'high', 'medium'}

# dummy화
le = LabelEncoder()
data['sales'] = le.fit_transform(data['sales'])
data['salary'] = le.fit_transform(data['salary'])
# print(set(data['sales']))
# print(data['salary'][:3])       # low:1 medium : 2, high: 3

print(data.corr())

feature = data.iloc[:, 0:-1]
label = data.iloc[:, -1]

# train / test split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=30)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (10499, 9) (4500, 9) (10499,) (4500,)

sc = StandardScaler() 
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

model = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=30, n_jobs=1)
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총 개수:%d, 오류 수 :%d'%(len(y_test), (y_test != y_pred).sum()))  
print('%.5f'%accuracy_score(y_test, y_pred))
print('test : ', model.score(x_test, y_test))  
print('train : ', model.score(x_train, y_train))

# 중요 변수 확인
print('특성(변수) 중요도 : {}'.format(model.feature_importances_))
# 중요 변수 시각화
def plot_feature_importances(model):
    n_features = feature.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.xlabel('attr importance')
    plt.ylabel('attr')
    plt.yticks(np.arange(n_features), feature.columns)
    plt.show()

plot_feature_importances(model)
# average_monthly_hours, last_eval‎uation, satisfaction_level << 중요 변수 3개

real_train = np.array(x_train[:, [0,1,3]])
real_test = np.array(x_test[:, [0,1,3]])

# randomforest
model.fit(real_train, y_train)
y_pred = model.predict(real_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('%.5f'%accuracy_score(y_test, y_pred))            # 0.58867

# deep learning
kmodel = Sequential([
    Dense(units=128, input_shape=(3,), activation='relu'),
    Dropout(rate=0.2),
    Dense(units=64, activation='relu'),
    Dropout(rate=0.2),
    Dense(units=32, activation='relu'),
    Dropout(rate=0.2),
    Dense(units=3, activation='softmax'),
])

print(kmodel.summary())


print(real_train.shape)

kmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = kmodel.fit(real_train, y_train, epochs=100, batch_size=128, validation_split=0.2 , verbose=2)

# 분류 예측
y_pred2 = kmodel.predict(real_test)
print('예측값 : ', np.argmax(y_pred2[:1]))
print('실제값 : ', y_test[:1])
loss, acc = kmodel.evaluate(real_test, y_test, batch_size=32)
print('acc : ',acc )    # acc :  0.48755

# randomforest가 분류 정확도가 더 좋다.





