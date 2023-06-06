# 다층 퍼셉트론(Multi-layer Perceptron,MLP)는 퍼셉트론으로 이루어진 층(layer) 여러개를 순차적으로 붙여놓은 형
# MLP는 정방향 인공신경망(feed-forward deep neural network,FFDNN)이라고 부르기도 함.
# deep learning은 완전 연결층, 병렬 연산한다.
# MLP는 CPU로 하면 성능이 떨어지기 때문에 GPU가 필요하다.
# MLP는 최근에 Deep Learning으로 이름이 변경되었다.

# 논리회로 분류 (and, or, xor)
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0,0], [0,1], [1,0], [1,1]])
print(feature)
# label = np.array([0,0,0,1])     # and
# label = np.array([0,1,1,1])     # or
label = np.array([0,1,1,0])     # xor. Perceptron은 노드 한개일 경우 xor을 해결 할 수 없다.

ml2 = MLPClassifier(hidden_layer_sizes=10 ,max_iter=100, activation='relu', solver='adam',\
                    learning_rate_init = 0.1, verbose=1).fit(feature, label)
# activation='relu': activation function 설정 , solver='adam': cost를 최소화 하는 function 선택., learning_rate_init : 학습률.  
# hidden_layer_sizes : 노드의 개수
# 학습하다가 굳이 더 학습할 필요가 없다고 판단되면 자동으로 학습을 멈춘다.
print(ml2)
pred2 = ml2.predict(feature)
print(pred2)
print(accuracy_score(label, pred2))


