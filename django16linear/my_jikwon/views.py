# 장고로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
# LinearRegression 사용. Ajax 처리!!!

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
from datetime import datetime
from django.http.response import HttpResponse, JsonResponse
import joblib
from my_jikwon.models import Jikwon

# Create your views here.

flag = False

def mainFunc(request):
    global flag
    
    if flag == False:
        makeModel()
        flag = True
        
    return render(request, 'show.html')

def makeModel():
# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
    datas = Jikwon.objects.values('jikwon_ibsail', 'jikwon_pay','jikwon_jik').all()
    jikdf = pd.DataFrame.from_records(datas)
    # print(jikdf.head(3), len(jikdf))
    # 근무년수
    for i in range(len(jikdf['jikwon_ibsail'])):
        jikdf['jikwon_ibsail'][i] = int((datetime.now().date() - jikdf['jikwon_ibsail'][i]).days / 365)
        
    jikdf.columns = ['근무년수', '연봉', '직급']
    # print(jikdf.head(2))
    print(type(jikdf['연봉'][3]))
    # train / test split (8:2)
    train_set, test_set = train_test_split(jikdf, test_size=0.2, random_state=12)
    print(train_set.shape, test_set.shape)      # (24, 3) (6, 3)
    
    model_lm = LinearRegression().fit(X=train_set.iloc[:,[0]], y=train_set.iloc[:,[1]])
    
    # 성능확인
    test_pred = model_lm.predict(test_set.iloc[:,[0]])
    print('예측값 : ', test_pred[:5].flatten())    # [6827.05696203 4767.56329114 5282.43670886 3737.8164557  7341.93037975]
    print('실제값 : ', test_set.iloc[:,[1]][:5].values.flatten())  # [7800 6600 5500 4000 8800]
    
    global r2s
    r2s = r2_score(test_set.iloc[:,[1]], test_pred)
    print('결정계수 (설명력) : ', r2s)     # 0.552302    << 양호한 수치
    
    # 모델 저장. pickle or joblib로 저장할 수 있다.
    joblib.dump(model_lm, "C:/work/psou2/django16linear/my_jikwon/static/jikmodel.model")   # django16linear/my_jikwon/static/jikmodel.model << 경로를 이렇게 잡아줘도 된다.
    
    # 직급별 연봉 평균
    global pay_jik
    # pay_jik = jikdf.groupby('직급').mean().round(3)
    pay_jik = jikdf.groupby('직급').mean().round(3)
    print('pay_jik', pay_jik)
    
    
# 임의의 함수에서 csrf를 적용하고 싶지 않은 경우에 적용시키면 된다. csrf token 적용 안한다고 선언.
@csrf_exempt    
def predictFunc(request):
    year = request.POST['year']
    # print(year)
    new_year = pd.DataFrame({'근무년수':[year]})
    
    # 모델을 읽어 해당 년도의 연봉을 예측하여 클라이언트에게 전송
    model = joblib.load("django16linear/my_jikwon/static/jikmodel.model")
    
    new_pred = round(model.predict(new_year)[0][0], 2)
    print(new_pred)
    
    context = {'new_pred':new_pred, 'r2s':r2s, 'pay_jik':pay_jik.to_html()}

    # return HttpResponse(json.dumps(context), content_type='application/json')
    return JsonResponse(context)



