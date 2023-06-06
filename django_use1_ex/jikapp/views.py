from django.shortcuts import render

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jikapp.models import Jikwon
import datetime
plt.rc('font', family='malgun gothic')

# 1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
#       : 부서번호, 직원명 순으로 오름 차순 정렬 
#   2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.
#   3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
#   4) 성별, 직급별 빈도표를 출력하시오.
# Create your views here.

def mainFunc(request):
    data = Jikwon.objects.select_related('buser_num').all().order_by('buser_num', 'jikwon_name')
    # data : dict 타입
    dt = datetime.datetime.now()
    re_dt = dt.strftime("%Y")
    print(re_dt)
    
    datas = []
    for j in data:
        dic = {"jikwon_no":j.jikwon_no , "jikwon_name":j.jikwon_name, "buser_name":j.buser_num.buser_name, \
               'jikwon_jik':j.jikwon_jik, 'jikwon_pay':j.jikwon_pay ,'jikwon_ibsail':(int(re_dt)-int(j.jikwon_ibsail.strftime("%Y"))),\
                'jikwon_gen':j.jikwon_gen}
        datas.append(dic)
    
    df = pd.DataFrame(datas)
    df.columns = ['사번', '직원명', '부서명', '직급', '연봉', '근무년수', '성별']
    # print(df.head(3))
    
    pdf = df.pivot_table(index=['부서명','직급'], values='연봉', aggfunc={np.mean,np.sum})
    
    pdf2 = df.pivot_table(index='부서명', values='연봉', aggfunc={np.mean,np.sum})
    pdf2.plot(kind='bar')
    
    fig = plt.gcf()
    fig.savefig('C:/work/psou2/django_use1_ex/jikapp/static/images/test1.jpg') 
    
    crotab = pd.crosstab(df["성별"], df['직급'])
    
    context = {'data':df.to_html(), 'data2':pdf.to_html(), 'data3':crotab.to_html()}
    return render(request, 'main.html', context)