from django.shortcuts import render

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jikapp.models import Jikwon
plt.rc('font', family='malgun gothic')
import sys
# 1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
#       : 부서번호, 직원명 순으로 오름 차순 정렬 
#   2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.
#   3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
#   4) 성별, 직급별 빈도표를 출력하시오.
# Create your views here.
def mainFunc(request):
    datas = Jikwon.objects.all().extra(
        tables=['buser'], 
        where=['buser.buser_no=buser_num'], 
        select={'buser_name': 'buser.buser_name'})
    df = pd.DataFrame.from_records(data= datas)
    print(df)