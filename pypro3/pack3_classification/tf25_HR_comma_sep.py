# 문제4) testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.
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

# 조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
# Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#data load
dataset = pd.read_csv('../testdata/HR_comma_sep.csv')
x_feature = dataset[dataset.columns.difference(['salary'])].copy()
for column in x_feature.columns:
    x_feature[column] = LabelEncoder().fit_transform(x_feature[column])
y_label = LabelEncoder().fit_transform(dataset['salary'])

x_train, x_test, y_train, y_test = train_test_split(x_feature,y_label,test_size=0.3,random_state=42)
''
#RandomForestClassifier
fmodel = RandomForestClassifier(criterion='entropy',n_estimators=100).fit(x_train,y_train)
# 중요도 추출
importance = fmodel.feature_importances_
# 중요도 시각화
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(x_feature.columns, importance)
plt.show() 

#확인된 상위3개의 중요변수
#average_montly_hours / last_evaluation / satisfaction_level

#======keras 분석======

from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input
from keras.models import Model

#중요변수 상위3개 feature 재 지정
importance_feature = ['average_montly_hours','last_evaluation','satisfaction_level']
importance_x_train = x_train[importance_feature]
importance_x_test = x_test[importance_feature]

#train/test label encoding
nb_classes = 3
y_train_onehot = to_categorical(y_train,nb_classes)
y_test_onehot = to_categorical(y_test,nb_classes)

# ============
'''
#EarlyStopping
es = EarlyStopping(patience=10,mode='auto',monitor='val_loss')

#ModelCheckpoint 
MODEL_DIR = '../mymodel/'
modelPath = 'HR_comma_sep_model.hdf5'
chkPoint = ModelCheckpoint(filepath=MODEL_DIR + modelPath, \
                           monitor='val_loss',verbose=0,save_best_only=True)

#Functional model
inputs = Input(shape=(3,)) #입력층
output1 = Dense(units=32,activation='relu')(inputs) #은닉층  
output2 = Dense(units=64,activation='relu')(output1) #출력층
output3 = Dense(units=32,activation='relu')(output2) #은닉층  
output4 = Dense(units=nb_classes,activation='softmax')(output3) #은닉층  

model = Model(inputs,output4)

model.compile(optimizer = 'rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(importance_x_train, y_train_onehot,epochs=999,batch_size=32,verbose=2,\
                    validation_split=0.3,callbacks=[es,chkPoint])

#시각화
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()
'''
# ============

#load
from keras.models import load_model
loadmodel = load_model('../mymodel/HR_comma_sep_model.hdf5')

#평가 RandomForestClassifier / Functional 
loss,acc = loadmodel.evaluate(importance_x_test,y_test_onehot,batch_size=32,verbose=2)
print('Functional loss : {:.3f}, accuracy : {:.3f}'.format(loss,acc))

from sklearn.metrics import accuracy_score
print('RandomForestClassifier accuracy: {:.3f}'.\
      format(accuracy_score(y_test,fmodel.predict(x_test))))

