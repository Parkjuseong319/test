# LFW (Labeled Faces in the Wild) 데이터
# 유명 정치인 등의 실제 얼굴에 대한 이미지 데이터. fetch_lfw_people() 명령으로 로드한다.
# 해상도는 50x37=5,828픽셀이고 각 채널이 0~255의 정수값을 가지는 컬러 이미지이다. 5,749명의 13,233개 사진을 가지고 있다. 
# 다음 인수를 사용하여 로드하는 데이터의 양과 종류를 결정할 수 있다.

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline      # 여러개의 명령을 차례대로 쭉 진행할때 사용된다.
from matplotlib.pyplot import xlabel
from sklearn.metrics._classification import classification_report

faces = fetch_lfw_people(min_faces_per_person=60, color=False)   # 각 인물 당 사진 최소 60개
# print(faces)
# print(faces.DESCR)
print(faces.data)       
print(faces.data.shape) # (1348, 2914)
print(faces.target)
print(faces.target_names)
print(faces.images.shape)

print()
# print(faces.images[50])
# print(faces.target_names[faces.target[50]])
# plt.imshow(faces.images[50], cmap='bone')
# plt.show()

# fig, ax = plt.subplots(3, 5)
# for i,axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
# plt.show()

# 주성분 분석으로 이미지 차원을 축소한 후 분류모델 작업 진행
m_pca = PCA(n_components = 150, whiten=True, random_state=0)   # whiten=True 주성분 스케일이 작아지도록 조정
x_low = m_pca.fit_transform(faces.data)
print('x_low : ', x_low, ' ', x_low.shape)      # (1348, 62, 47) -> (1348, 150). 3차원 2차원으로 축소

m_svc = SVC(C=100)
model = make_pipeline(m_pca, m_svc)     # 선처리기(PCA)와 분류기를 하나의 파이프라인으로 만들고 순차적으로 실행
print(model)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model.fit(x_train, y_train)     # image와 label 학습 
pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10])

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
mat = confusion_matrix(y_test, pred)
print('confusion_matrix : \n', mat)
print('accuracy : ', accuracy_score(y_test, pred))      # 0.8397626

print(classification_report(y_test, pred, target_names=faces.target_names))
# weighted avg : 가중상수 평균

fig, ax = plt.subplots(4, 6)
for i,axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[pred[i]].split()[-1], color='black' if pred[i] == y_test[i] else 'red')
    
plt.show()

# 오차행렬로 시각화
import seaborn as sns

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,\
             xticklabels=faces.target_names, yticklabels=faces.target_names)      # fmt = 'd'는 decimal
plt.xlabel('true(real) lable')
plt.ylabel('predicted lable')
plt.show()






