# 랜덤 포레스트 (Random Forest) 분류모형
# 랜덤 포레스트는 분류, 회귀분석 등에 사용되는 앙상블 학습방법의 일종으로, 훈련 과정에서 구성한
# 다수의 결정 트리로 부터 분류 또는 평균 예측치(회귀 분석)를 출력함으로써 동작한다.
# • Decision Tree에 비해 높은 정확성, 불완전성을 제거 
# • 간편하고 빠른 학습 및 테스트 알고리즘
# • 변수 소거 없이 수천 개의 입력 변수들을 다루는 것이 가능
# • 임의화를 통한 좋은 일반화 성능
# • 다중 클래스 알고리즘 특성
# 앙상블 기법의 일종으로 Bagging을 사용한다.
import numpy as np
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv")
print(df.head(2), df.shape)
print(df.isnull().any())        # Age, Cabin, Embarked에 null값 존재
df = df.dropna(subset=['Pclass', 'Age','Sex', 'Survived'])
print(df.shape)     # (714, 12)


df_x = df[['Pclass', 'Age','Sex']]
print(df_x.head(2))
df_y = df['Survived']
print(df_y.head(2))

# Scaling(성별을 수치로 Dummy 변수화)
# df_x['Sex'] = df_x['Sex'].apply(lambda x:1 if x == 'male' else 0)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df_x.loc[:, 'Sex'] = LabelEncoder().fit_transform(df_x["Sex"])      # apply 대신 이렇게 사용도 가능하다. 문자열을 수치로 변환 가능.

print(df_x.head(2))
"""
# 참고 - Pclass를 원핫 처리
print(set(df_x['Pclass']))  # {1, 2, 3}
df_x2 = pd.DataFrame(OneHotEncoder().fit_transform(df_x['Pclass'].values[:, np.newaxis]).toarray(), columns=['f', 's', 't'])
df_x = pd.concat([df_x, df_x2], axis=1)
print(df_x)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x,df_y, test_size=0.3, random_state=12)
print(train_x[:2])
print(train_y[:2])

model = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(train_x, train_y)
# ramdom forest는 랜덤하게 샘플링하기 때문에 분류 정확도같은 수치들이 조금씩 변한다.
pred = model.predict(test_x)
print("예측값 : ", pred[:10])
print("실제값 : ", np.array(test_y[:10]))

# 분류 정확도
print('acc : ', sum(test_y == pred) / len(test_y))
from sklearn.metrics import accuracy_score
print('acc2 : ', accuracy_score(test_y, pred))

# 교차검증(K-Fold)
from sklearn.model_selection import KFold, cross_val_score
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print(cross_vali)
print('cross_vali : ', np.around(np.mean(cross_vali),3))

# 중요 변수 확인
print('특성(변수) 중요도 : {}'.format(model.feature_importances_))
# 중요 변수 시각화
import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = df_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.xlabel('attr importance')
    plt.ylabel('attr')
    plt.yticks(np.arange(n_features), df_x.columns)
    plt.show()

plot_feature_importances(model)


