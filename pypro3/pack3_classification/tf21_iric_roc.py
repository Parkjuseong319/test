# iris dataset으로 다항분류 모델 작성 : ROC curve로 성능 출력
# layer 1개, 2개, 3개, ...처럼 layer가 다른 모델을 여러 개 만들어 성능 비교
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

iris = load_iris()
print(iris.keys())
x = iris.data
y = iris.target
print(x[:1])
print(y[:1])

names = iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# label one-hot
onehot = OneHotEncoder(categories = 'auto')
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y[:2], y.shape)

# feature에 대한 표준화 / 정규화 시행 - 대개의 경우 시행했을 때 더 좋은 결과를 얻음
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:1])

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
n_features = x_train.shape[1]   # 4
n_classes = y_train.shape[1]    # 3

# model
from keras.models import Sequential
from keras.layers import Dense

def create_model_func(input_dim, output_dim, out_nodes, n , model_name='model'):
    # print(input_dim, output_dim, out_nodes, n , model_name)
    def create_model():
        model = Sequential(name = model_name)
        for _ in range(n):
            model.add(Dense(units=out_nodes, input_dim=input_dim, activation='relu'))
        model.add(Dense(units=output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model
    return create_model     # 클로저 함수

models = [create_model_func(n_features, n_classes, 10, n ,'model_{}'.format(n)) for n in range(1,4)]
print(len(models))

# for cre_model in models:
#     print()
#     cre_model().summary()
    
history_dict = {}    
    
for cre_model in models:
    model = cre_model()
    print('model name : ', model.name)
    historys = model.fit(x_train, y_train, batch_size=5, epochs= 50, verbose=0, validation_split=0.3)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('test loss : ', score[0])
    print('test accuracy : ', score[1])
    history_dict[model.name] = [historys, model]
    
print(history_dict)

# 시각화 
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize = (8,6))

for model_name in history_dict:
    print('h_d : ', history_dict[model_name][0].history['acc'])
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('validation accuracy')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
    
plt.show()
    
# 분류 모델에 대한 성능 평가 기법 중 하나로 ROC curve / AUC 출력
from sklearn.metrics import roc_curve, auc

plt.figure()
plt.plot([0,1], [0,1], 'k--')

for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel() )
    plt.plot(fpr, tpr, label='{}, AUC value : {:.3f}'.format(model_name, auc(fpr, tpr)))
    
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

    




