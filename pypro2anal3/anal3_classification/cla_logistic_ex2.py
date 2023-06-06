# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값0(착용X), 값1(착용O)
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('../testdata/bodycheck.csv')
print(data.head(3))
print(data.loc[:,['게임','TV시청', '안경유무']].corr())      # 0.795932

x = data.loc[:,['게임','TV시청']]
y = data['안경유무'].values

# 과적합 방지를 목적으로 train(모델 학습용) / test(모델 검정용)로 데이터를 분리 (7:3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     # (14, 2) (6, 2) (14,) (6,)

# 모델 생성
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0)
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총 개수:%d, 오류 수 :%d'%(len(y_test), (y_test != y_pred).sum()))  # 총 개수:6, 오류 수 :1
print('%.5f'%accuracy_score(y_test, y_pred))    # 0.83333

print('test : ', model.score(x_test, y_test))   
print('train : ', model.score(x_train, y_train))

# 새로운 데이터(키보드로 입력)로 분류 확인
input_data = pd.DataFrame({'게임':[int(input("게임 시간 입력해봐 : "))], 'TV시청':[int(input("TV시청 시간 입력해봐 : "))]})
print('안경씀' if np.rint(model.predict(input_data))[0] == 1 else '안경 안씀')


