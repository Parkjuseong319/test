# BMI 식을 사용해 대량의 dataset 만든 후 모델에 적용하기
# BMI를 이용한 비만도 계산은 자신의 몸무게를 키의 제곱으로 나누는 것으로 공식은 kg/㎡.
# BMI가 18.5 이하면 저체중 ／ 18.5 ~ 22.9 사이면 정상 ／ 23.0 ~ 24.9 사이면 과체중 ／ 25.0 이상부터는 비만으로 판정.
# ex) 키 170㎝에 몸무게 73kg이면, 계산식 : 73 / (1.7×1.7) = 25.26 → 과체중
"""
import random
random.seed(12)

def calc_bmi(h, w):
    bmi = w / (h/100)**2
    # print(bmi)
    if bmi < 18.5: return 'thin'
    elif bmi < 25.0: return 'normal'
    return 'fat'
    
print(calc_bmi(170, 73))
fp = open('bmi.csv', 'w')
fp.write('height,weight,label\n')

# 무작위로 데이터 생성
cnt = {'thin':0, 'normal':0, 'fat':0}

for i in range(50000):
    h = random.randint(150, 200)
    w = random.randint(35, 100)
    label = calc_bmi(h, w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h,w,label))
    
fp.close()
"""
# SVM으로 분류 모델 작성
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

tbl = pd.read_csv('bmi.csv')
print(tbl.head(2), tbl.shape)
print(tbl.info())

label = tbl['label']

# 정규화 : 0 ~ 1 사이의 값으로 변환 시키는 것.
w = tbl['weight'] / 100
print(w[:3].values)
h = tbl['height'] / 200
print(h[:3].values)

# label을 dummy 자료로 만들어주기
print(label[:3])
label = label.map({'thin':0, 'normal':1, 'fat':2})
print(label[:3])
wh = pd.concat([w,h], axis =1)

x_train, x_test, y_train, y_test = train_test_split(wh, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     # (35000, 2) (15000, 2) (35000,) (15000,)

model = svm.SVC(C=100).fit(x_train, y_train)              # C 숫자가 클수록 과적합 방지가 잘된다.
print(model)

pred = model.predict(x_test)
print('예측 -',pred[:10])
print('실제 -', y_test[:10].values)

acc = metrics.accuracy_score(y_test,pred)
print('분류 정확도 : ', acc)     # 0.99093

# 시각화
tbl2 = pd.read_csv('bmi.csv', index_col=2)
print(tbl2.head(3))

def plot_func(lbl, color):
    b = tbl2.loc[lbl]
    plt.scatter(b['weight'], b['height'], c=color, label=lbl)
    
plot_func('fat', 'red')
plot_func('normal', 'green')
plot_func('thin', 'blue')
plt.legend()
plt.show()

# 새로운 값으로 예측
new_data = pd.DataFrame({"weight":[56,88], 'height':[170, 170]})
new_data['weight'] = new_data['weight'] / 100 
new_data['height'] = new_data['height'] / 200 
new_pred = model.predict(new_data)
print('새로운 값 예측 결과 : ', new_pred)






