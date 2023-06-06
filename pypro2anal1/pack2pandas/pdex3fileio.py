# file i/o
import pandas as pd
import numpy as np

df = pd.read_csv("../testdata/ex1.csv")
print(df, type(df))     # type은 dataframe
print()
df = pd.read_table("../testdata/ex1.csv", sep=',')
print(df)
print()
df = pd.read_csv("../testdata/ex2.csv", header=None)        # header=None을 주면 헤더가 없는 dataframe에 임의의 헤더 부여(보통 번호)
print(df)
print()
df = pd.read_csv("../testdata/ex2.csv", names=['a','b'])    # 칼럼명을 주면 오른쪽 부터 채운다.
print(df)
print()
df = pd.read_csv("../testdata/ex2.csv", names=['a','b','c','d','e'])    # 칼럼명을 주면 오른쪽 부터 채운다.
print(df)
print()
df = pd.read_csv("../testdata/ex3.txt")
print(df)
print(df.info())
print()
# df = pd.read_table("../testdata/ex3.txt", sep='\s+')    # 정규표현식으로 값 구분
df = pd.read_table("../testdata/ex3.txt", sep='\s+', skiprows=[1,3])    # skiprows 속성으로 특정 행을 표시 안할 수 있다.
print(df)
print()
result = pd.read_fwf("../testdata/data_fwt.txt", header=None, encoding='UTF-8', widths=(10,3, 5), names=('날짜', '이름', '매출액'))
# widths로 길이 구분지었다.
print(result)

print()
# chunk 단위로 데이터 읽기
test = pd.read_csv("../testdata/data_csv2.csv", header=None, chunksize=3)
# chunksize 속성에 준 만큼 끊어놓는다.
print(test)
for piece in test:
    # print(piece)
    print(piece.sort_values(by=2, ascending=True))  # 2열을 기준으로 ascending한다.

print('-----'*7)
items = {
        "apple":{'count':10, 'price':1500},
        "orange":{'count':25, 'price':1000},
    }
df = pd.DataFrame(items)
print(df)
# df.to_clipboard()    # clipboard에 저장된다.
print(df.to_csv())      # csv 형태가 된다.
print(df.to_json())
print(df.to_html())

df.to_csv('pdex3_1.csv', sep=",")
df.to_csv('pdex3_2.csv', sep=",", index=False)
df.to_csv('pdex3_3.csv', sep=",", index=False, header=False)

print()
# writer = pd.ExcelWriter('pdex3excel.xlsx', engine='xlsxwriter')
# df.to_excel(writer, sheet_name='sheet1')
# writer.save()

print()
# Excel file 읽기
myexcel = pd.ExcelFile('pdex3excel.xlsx')
df2 = myexcel.parse(sheet_name="sheet1")
print(df2)
print()
df3 =pd.read_excel("pdex3excel.xlsx", sheet_name='sheet1')
print(df3)

