# 분류분석 - Decision Tree(CART) : 정보획득 지표(entropy, gini)가 작아지도록 분류. 
# 즉 불순도가 없을 때까지 계속적으로 이진 분류를 진행. 알고리즘은 간단하나 성능은 우수.
from sklearn import tree

# 키와 머리카락 길이로 남여 구분
x = [[180, 15], [177, 52], [156, 35], [175, 5], [166, 22],
     [187, 10], [167, 2], [156, 25], [175, 35], [173, 15]]
y = ['man', 'woman', 'woman', 'man', 'man', 'man', 'woman', 'man', 'man','man']
label_names = ['height', 'hair_length']

model = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
print(model)
model.fit(x, y)
print('분류 정확도(predict X) : ', model.score(x, y))
pred = model.predict(X=x)
print('예측결과 : ', pred)

#분류 과정 시각화
import pydotplus
import collections
dot_data = tree.export_graphviz(model,feature_names=label_names,out_file=None,filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors =('red','orange')
edges = collections.defaultdict(list)       #list type의 변수를 준비

for e in graph.get_edge_list(): 
    edges[e.get_source()].append(int(e.get_destination()))
    
for e in edges:
    edges[e].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[e][i]))[0]
        dest.set_fillcolor(colors[i])
graph.write_png('tree.png')         #이미지로 저장

# 이미지 읽기
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
img = imread('tree.png')
plt.imshow(img)
plt.show()

# 새로운 값으로 예측
new_pred = model.predict([[170,12], [160,40]])
print('예측결과 : ',new_pred)



