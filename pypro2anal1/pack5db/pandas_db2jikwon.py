# 원격 DB 연동 후 jikwon 자료를 읽어 DataFrame에 저장

import MySQLdb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import sys
import csv

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
        select jikwon_no, jikwon_name, jikwon_jik, buser_name, jikwon_gen, jikwon_pay from jikwon
        inner join buser on buser_num=buser_no
    """
    cursor.execute(sql)     #  on buser_num=buser_no 이 기준키가 되기 때문에 부서명이 group by 된 상태로 출력된다.
    
    # 출력1 : console 출력
    for (jikwon_no, jikwon_name, jikwon_jik, buser_name, jikwon_gen, jikwon_pay) in cursor:
        print(jikwon_no, jikwon_name, jikwon_jik, buser_name, jikwon_gen, jikwon_pay)
    
    print()
    
    # 출력2 : DataFrame
    df1 = pd.DataFrame(cursor.fetchall(), \
                       columns=['jikwon_no', 'jikwon_name', 'jikwon_jik', 'buser_name', 'jikwon_gen', 'jikwon_pay'] )
    print(df1.head(3))
    print()
    
    # 출력3 : csv 파일로 저장 후 읽기
    with open('jik_data.csv', mode='w', encoding='UTF-8') as obj2:
        writer = csv.writer(obj2)
        for r in cursor:
            writer.writerow(r)
    
    df2 = pd.read_csv('jik_data.csv', header=None, names=['사번', '이름', '직급', '부서명', '성별', '연봉'])
    print(df2.head(4))
    print()
    
    # 출력4 : DataFrame sql 사용
    df = pd.read_sql(sql, conn)
    df.columns = ['사번', '이름', '직급', '부서명', '성별', '연봉']
    print(df.head(3))
    print('-------' * 7)
    print(df[:3])
    print(df[:-27])     # 뒤에서부터 27개를 제외한 행들을 출력
    print('건수 : ', len(df))
    print('건수 : ', df['이름'].count())
    print('직급별 인원 수 : ', df['직급'].value_counts())   # value_counts는 group by 컬럼명 having count()과 같음
    print('연봉 평균 : ', df.loc[:,'연봉'].mean())
    print('연봉 표준편차 : ', df.loc[:,'연봉'].std())
    print(df.loc[:,'연봉'].describe())
    print(df.loc[df['연봉'] >= 8000])
    
    # 교차표. 범주형 데이터에 관해서만 사용 가능
    ctab = pd.crosstab(df["성별"], df['직급'], margins=True)
    print(ctab)     # ctab.to_html() 도 가능하다.
    
    print()
    print(df.groupby(['성별', '직급'])['이름'].count())
    print(df.pivot_table(['연봉'], index=['성별', '직급'], aggfunc= np.mean))
    
    # 시각화
    jik_ypay = df.groupby(['직급'])['연봉'].mean()
    print(jik_ypay)
    print(jik_ypay.index)
    print(jik_ypay.values)
    
    plt.pie(jik_ypay, explode=(0.1,0,0,0.2,0), labels=jik_ypay.index, shadow=True, labeldistance=0.6,\
            counterclock=False)      # counterclock True는 반시계, False는 시계 방향. shadow는 그림자. 
    plt.show()
    
except Exception as e:
    print("handler error ", e)
    cursor.close()
    conn.close()
