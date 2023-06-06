from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

train = [
    [5,3,2], 
    [1,3,5], 
    [4,5,7] 
]
label = [0, 1, 1]

plt.plot(train, 'o')
plt.xlim([-1, 5])
plt.ylim([0, 10])
plt.show()

kmodel = KNeighborsClassifier(n_neighbors=3, weights='distance')    # KNN은 적절한 k 값을 주는것이 중요하다.
kmodel.fit(train, label)
pred = kmodel.predict(train)
print('pred : ', pred)
print('test 정확도 : {:.3f}'.format(kmodel.score(train, label)))

new_data = [[1,2,8], [6,2,1]]
new_pred = kmodel.predict(new_data)
print('new_pred : ', new_pred)
