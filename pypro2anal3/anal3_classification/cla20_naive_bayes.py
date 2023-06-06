# *조건부 확률 *
# P(B|A) : 사건 A에 대해서,사건 A가 발생했을 때 사건 B가 발생할 확률. “사건 A에 대한 사건 B의 조건부 확률”
# 이라 한다. 예를 들어 어느 집단의 남학생일 확률이 P(A)라고 하고,학생의 키가 170이 넘는 확률을 P(B)라고
# 했을 때, 남학생 중에서 키가 170이 넘는 확률은 B의 조건부 확률이 되며 P(B|A)로 표현한다.
# - 사전확률 : 특정 사건이 일어나기 전의 확률로 베이즈 추론에서 관측자가 관측을 하기 전에 가지고 있는 확률분포를 말한다.
# - 사후확률 : 확률변수 관측에 대한 조건부 확률로, 어떤 사건이 발생하였고, 이 사건이 나온 이유가 무엇인지 P(B|A)란
# 식으로 나타낸 것이다. (B는관측한 사건. A는 B가 나올 수 있게 한 과거의 사실)
# - 베이즈 정리 (ML에서는 귀납적 추론방식을 주로 사용) : 두 확률변수의 사전확률과 사후확률 사이의 관계를 나타내는 정리다. 
#     베이즈 정리는 사전확률로부터 사후확률을 구할 수 있는 개념이다.
# P(L|Features) = (P(Features | L ) * P(L)) / P(Features)
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder     # deep learning에서는 필수적

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 3, 5, 7, 9])

model = GaussianNB().fit(x,y)
print(model)
pred = model.predict(x)
print(pred)
print('분류정확도 : ',metrics.accuracy_score(y, pred))

new_x = np.array([[0.5], [2.3], [7], [15]])
new_pred = model.predict(new_x)
print(new_pred)

print('feature를 OneHotEncoder 처리 후 모델 작성 --------')
x = np.eye(len(x))      # 방법1 
print(x)
model = GaussianNB().fit(x,y)
print(model)
pred = model.predict(x)
print(pred)

print()     # 방법2
one_hot = OneHotEncoder(categories = 'auto')
x = np.array([[1], [2], [3], [4], [5]])
x = one_hot.fit_transform(x).toarray()
print(x)


