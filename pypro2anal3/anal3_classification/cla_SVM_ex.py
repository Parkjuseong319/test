"""
[SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
https://www.kaggle.com/zhaoyingzhu/heartcsv
https://github.com/pykwon/python/tree/master/testdata_utf8         Heart.csv

Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
임의의 값을 넣어 분류 결과를 확인하시오.     
정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ

feature 칼럼 : 문자 데이터 칼럼은 제외
label 칼럼 : AHD(중증 심장질환)

데이터 예)
"","Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"
"1",63,1,"typical",145,233,1,2,150,0,2.3,3,0,"fixed","No"
"2",67,1,"asymptomatic",160,286,0,2,108,1,1.5,2,3,"normal","Yes"
...
"""
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split


df = pd.read_csv('../testdata/heart.csv', index_col=0)
print(df.head(3))
# print(df.info())
"""
0   Unnamed: 0  303 non-null    int64  
 1   Age         303 non-null    int64  
 2   Sex         303 non-null    int64  
 3   ChestPain   303 non-null    object 
 4   RestBP      303 non-null    int64  
 5   Chol        303 non-null    int64  
 6   Fbs         303 non-null    int64  
 7   RestECG     303 non-null    int64  
 8   MaxHR       303 non-null    int64  
 9   ExAng       303 non-null    int64  
 10  Oldpeak     303 non-null    float64
 11  Slope       303 non-null    int64  
 12  Ca          299 non-null    float64
 13  Thal        301 non-null    object 
 14  AHD         303 non-null    object 
 """
# ChestPain, Thal은 문자데이터로 되어있으므로 feature에서 제외한다.
print(df.isnull().any())        # Ca, Thal column에서 null값 확인
print(df.shape)     # (303, 14)
df = df.dropna(subset=('Thal'))
df['Ca'] = df["Ca"].fillna(df["Ca"].mean())     # 결측치 해당 칼럼 평균값으로 채워줌
print(df.shape)     # (301, 14)

feature = df.drop(columns=['ChestPain', 'Thal', 'AHD'])
label = df['AHD'].apply(lambda x:1 if x== 'Yes' else 0)
print(feature[:3])
print(label[:3])

# train / test split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=10)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (210, 12) (91, 12) (210,) (91,)

# 모델 생성
model = svm.SVC(C=100).fit(x_train, y_train)
pred = model.predict(x_test)
print('예측값 -', pred[:10])
print('실제값 -', y_test[:10].values)

print("분류 정확도 : ", metrics.accuracy_score(y_test, pred))        # 분류 정확도 :  0.7912

# 임의의 값으로 예측
new_data = pd.DataFrame({"Age":[70, 50],"Sex":[1, 0] ,"RestBP":[180,150],\
            "Chol":[260,200],"Fbs":[0,1],"RestECG":[2,2],"MaxHR":[100,145],\
            "ExAng":[1, 0],"Oldpeak":[2.5, 1.2],"Slope":[2,3],"Ca":[3, 0]})
new_pred = model.predict(new_data)
result = ""
for i in new_pred:
    if i == 1:
        result += 'yes '
    else:
        result += 'no '
print('새로운 값 예측 결과(중증 심장질환 여부) : ', result)

