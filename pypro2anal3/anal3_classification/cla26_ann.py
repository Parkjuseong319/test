# Neural Network
# 단층 신경망(뉴런 , 노드가 1개 사용) - Perceptron
# input(x)의 가중치 합에 대해 임계치를 기준으로 두 가지 output 중 한가지를 출력(분류-정성적), 한 개의 연속형 결과를 출력(예측-정량적) 

# 논리회로 분류 (and, or, xor)
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0], [0,1], [1,0], [1,1]])
print(feature)
# label = np.array([0,0,0,1])     # and
# label = np.array([0,1,1,1])     # or
label = np.array([0,1,1,0])     # xor. Perceptron은 노드 한개일 경우 xor을 해결 할 수 없다.

ml = Perceptron(max_iter=10000, eta0=0.1, verbose=1).fit(feature, label)       # eta0은 learning rate와 같다.
print(ml)
pred = ml.predict(feature)
print(pred)
print(accuracy_score(label, pred))

