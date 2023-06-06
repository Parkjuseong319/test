# local db + pandas
import pandas as pd
import sqlite3

sql = "create table if not exists test(product varchar(10), maker varchar(10), weight real, price integer)"
conn = sqlite3.connect(':memory:')       # database속성을 ':memory:'으로 지정하면 램에다가만 저장한다는것을 잊지 말자
conn.execute(sql)
conn.commit()

stmt = 'insert into test values(?,?,?,?)'
data1 = ('mouse', 'sam', 12.5, 5000)
data2= ('keyboard', 'lg', 67.5, 25000)
conn.execute(stmt, data1)
conn.execute(stmt, data2)
data3 = [('mouse2', 'sam', 12.5, 6000), ('keyboard2', 'lg', 67.5, 26000)]   # 여러 개 넣는 법
conn.executemany(stmt, data3)       # 여러개 넣을때
conn.commit()

cursor = conn.execute("select * from test")
rows = cursor.fetchall()
for a in rows:
    print(a)

# DB의 자료를 DataFrame에 저장
# DataFrame을 사용하면 to_html을 통해 장고 출력이 편안해지기도 하고, pandas에서 제공하는 함수들을 다양하게 사용가능하기 때문에 장점이 많다
df1 = pd.DataFrame(rows, columns=['product', 'maker','weight', 'price'])
print(df1)

print()
df2 = pd.read_sql("select * from test", conn)   # sql문과 연결객체
print(df2)
print(df2.to_html())
print(pd.read_sql('select count(*) as "건수" from test', conn))

print('--------'*7)
datas = {
    'product':['연필','볼펜','애플펜슬'],
    'maker':['steadtler','모나미','Apple'],
    'weight':[1.5, 6.5, 3.6],
    'price':[500, 1500, 120000]
}
frame = pd.DataFrame(datas)
print(frame)
frame.to_sql('test', conn, if_exists='append', index=False)     # 기존 test 테이블과 이어 붙여줬다.
# index 속성에 False를 넣은 이유는 기존 테이블에 데이터를 추가하는 것이기 때문에 인덱스를 새로 만들면 구조가 안맞을 수 있기 때문에 False
df3 = pd.read_sql("select * from test", conn)
print(df3)

# 데이터프레임.to_sql에 대해        to_sql(name, 연결객체, if_exists=?, index= True or False)
# name='테이블명' 이름으로 기존 테이블이 있으면 해당 테이블의 컬럼명에 맞게 데이터를 넣을 수 있음
# if_exists='append' 옵션이 있으면, 기존 테이블에 데이터를 추가로 넣음
# if_exists='fail' 옵션이 있으면, 기존 테이블이 있을 경우, 아무일도 하지 않음
# if_exists='replace' 옵션이 있으면, 기존 테이블이 있을 경우, 기존 테이블을 삭제하고, 다시 테이블을 만들어서, 새로 데이터를 넣음
# 이미 만들어진 테이블이 없으면, name='테이블명' 이름으로 테이블을 자동으로 만들고, 데이터를 넣을 수 있음
# 테이블이 자동으로 만들어지므로, 테이블 구조가 최적화되지 않아 자동으로 테이블 만드는 것은 추천하지 않음




