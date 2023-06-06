# MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
#      - DataFrame의 자료를 파일로 저장
#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
#      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성

import MySQLdb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pack5db.pandas_db2jikwon import jik_ypay
plt.rc('font', family='malgun gothic')
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
        
except Exception as e:
    print("connect error ", e)
    sys.exit()  # 프로그램 강제종료
    
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = """
        select jikwon_no, jikwon_name, buser_name, jikwon_pay, jikwon_jik from jikwon
        inner join buser on jikwon.buser_num=buser.buser_no order by jikwon_no
    """
    cursor.execute(sql)
    
    df = pd.read_sql(sql, conn)
    df.columns = ['사번', '이름', '부서명', '연봉', '직급']    # columns는 tuple로 줘도 상관 없다.
    df.index = range(1, 31)
    print(df.head(3))
    print('부서명별 연봉의 합 : ', df.groupby(['부서명'])['연봉'].sum())
    print('부서명별 최대연봉 : ', df.groupby(['부서명'])['연봉'].max())
    print('부서명별 최소연봉 : ', df.groupby(['부서명'])['연봉'].min())
    # 교차 테이블
    print(pd.crosstab(df['부서명'], df['직급']))
    
    # 직원별 담당 고객자료(고객번호, 이름, 전화) 표시. 없으면 'X'
    for i in range(0,len(df.index)):
        sql2 = """
            select gogek_no, gogek_name, gogek_tel 
            from gogek inner join jikwon on gogek.gogek_damsano = jikwon.jikwon_no 
            where jikwon_no = {} 
        """.format(str(df.index[i]))
        
        cursor.execute(sql2)  # result 용

        if cursor.rowcount == 0:    # 조건문에 맞는 고객데이터가 없을때
            print(df['사번'][i+1],df['이름'][i+1],":담당고객 X")
            print()
        else:
            print(df['사번'][i+1],df['이름'][i+1],":담당고객 정보")
            for (gogek_no,gogek_name,gogek_tel) in cursor:
                print(gogek_no,gogek_name,gogek_tel)
            print()
        
    # 부서명별 연봉의 평균으로 가로 막대 그래프
    jik_ypay = df.groupby(['부서명'])['연봉'].mean()
    plt.barh(jik_ypay.index, jik_ypay.values)
    plt.grid(visible=True)  # 격자 선 넣어줄 수 있다.
    plt.show()

    
except Exception as e:
    print("handler error ", e)
    cursor.close()
    conn.close()