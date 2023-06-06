# pandas : 2차원 구조, 데이터 분석용 자료구조 및 함수를 제공
# numpy 기능을 계승, 스프레드시트, SQL 조작기능, 시각화, 시계열 처리, 축약연산, file i/o,...
# Data Munging(or Data Wrangling) : 원자료(raw data)를 보다 쉽게 접근하고 분석할 수 있도록 데이터를 정리하고 통합하는 과정

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# Series : 일련의 객체를 저장할 수 있는 1차원 배열 구조. numpy와는 다르게 명시적으로 보여준다.
# obj = Series([3, 7, -5, 4])     # indexing 자동으로 하고 datatype을 보여준다.
# obj = Series((3, 7, -5, 4))
# obj = Series({3, 7, -5, 4})     # set type은 순서가 없기 때문에 사용 불가     
obj = Series([3, 7, -5, 4], index=['a', 'b', 'c', 'd'])     # index 명 변경 가능.
print(obj, type(obj))
print(obj.sum(), sum(obj), np.sum(obj))     # 각각 pandas, python, numpy의 합계. pandas의 sum은 내부적으로는 numpy sum이다.

print()
print(obj.values)
print(obj.index)

print('인덱싱/슬라이싱')
print(obj['a'], obj[0], obj[['a']])
print(obj[['a', 'b']])
print(obj['a':'c'])
print(obj[1:4])
print(obj[[2,1]])
print(obj > 0)
print('a' in obj)

print('dict type 관련')   # dict type은 순서가 없음. index 명을 수정할때 잘못 수정 될 수 있다.
names = {'mouse':5000, 'keyboard':15000, "monitor":230000}
obj2 = pd.Series(names, name=('상품가격',))
obj2.index = ['마우스', '키보드', '모니터']
print(obj2, ' ', type(obj2))    # key가 index가 된다.
# obj2.name = '상품가격'      # 이름 부여 가능
print(obj2)
print('--------------------------')
# DataFrame : 2차원 자료구조. 여러 개의 열로 구성, 여러 개의 Series가 있는 느낌.
df = DataFrame(obj2)
print(df)

data = {
        'irum':['홍길동','신기태', '유명한', '한가희'],
        'juso':('역삼동', '신당동', '역삼동', '신사동'),
        'nai':[23,33,32,25],
    }
frame = DataFrame(data)
print(frame, type(frame))
print(frame['irum'])
print(frame.irum)
print(type(frame.irum))     # 열 하나만 추출했기 때문에 Series 타입

print(DataFrame(data, columns = ['juso','irum','nai'])) # column 순서 바꾸기

# NaN(결측치)
frame2 = DataFrame(data, columns=['irum','nai', 'juso', 'tel'], index=['a','b','c','d'])     # 값이 없는 칼럼을 만들면 값으로 NaN이 들어간다.
print(frame2)

frame2['tel'] = '111-1111'      # 값 하나만 줬기 때문에 모든 값이 이걸로 치환됨
print(frame2)

val = Series(['222-2222', '333-3333'], index=['b','d'])     # 특정 행에만 값을 선택적으로 줄 수 있다.
frame2['tel'] = val
print(frame2)

print()
print(frame2.T)     # Transpose

print()
print(frame2.values)    # 2차원 배열로 출력된다.
print(frame2.values[0,1])   # 스칼라 값만 나옴
print(frame2.values[0:2])   # 2차원 배열로 출력

print('행 또는 열 삭제 --------')
frame3 = frame2.drop('d', axis=0)   # 행 삭제
print(frame3)

frame4 = frame2.drop('tel', axis=1)   # 열 삭제
print(frame4)

print()
print(frame2.sort_index(axis=0, ascending=False))   # 행기준 descending
print()
print(frame2.sort_index(axis=1, ascending=True))    # 열기준 ascending
print()
print(frame2.rank(axis=0))                          # 행기준으로 랭킹 매김.

print()
data = {
        'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
        'inwon':[23, 25, 10]
    }
fr = DataFrame(data)
print(fr)

result1 = Series([x.split()[0] for x in fr.juso])
result2 = Series([x.split()[1] for x in fr.juso])
print(result1)
print(result2)
print(result1.value_counts())

print()
# Series의 재색인
data = Series([1,3,2], index=[1,4,2])
print(data)

data2 = data.reindex((1,2,4))
print(data2)

print('재색인 시 NaN 채우기')
data3 = data2.reindex([0,1,2,3,4,5])
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], fill_value = 777)      # fill_value를 통해 NaN을 채울 수 있다.
print(data3)
print()
# data3 = data2.reindex([0,1,2,3,4,5], method = 'ffill')      # method='ffill'은 NaN을 바로 앞 행의 값으로 채운다.
data3 = data2.reindex([0,1,2,3,4,5], method = 'pad')        # ffill과 같음
print(data3)

print()
data3 = data2.reindex([0,1,2,3,4,5], method = 'bfill')      # method='bfill'은 NaN을 바로 다음 행의 값으로 채운다.
data3 = data2.reindex([0,1,2,3,4,5], method = 'backfill')
print(data3)

print('-----------')
# bool 처리
df = DataFrame(np.arange(12).reshape(4,3), index=['1월', '2월', '3월', '4월'], columns=['강남','강북','서초'])
print(df)
print(df['강남'])
print(df['강남'] > 3)
print(df[df['강남'] > 3])     # 조건을 묶어버리면 참인 값들만 출력된다.
print()
print(df < 3)
df[df < 3] = 0
print(df)

print("슬라이싱 관련 method(function) - loc() : 라벨지원, iloc() : 숫자지원")
print(df.loc['3월', :])
print(df.loc[:'2월'])    # 2월 이하 행
print(df.loc[:'2월', ['서초']])    # 2월 이하 행 서초 열
print()
print(df.iloc[2])   # 인덱스 0번을 기준으로 2번 행 출력
print(df.iloc[2, :])
print(df.iloc[:3])  # 3번행 미만
print(df.iloc[:3, 2])
print(df.iloc[:3, 1:3]) # 3번행 미만 1열, 2열
print(df.iloc[1:3, 1:3])

print('산술연산')
s1 = Series([1,2,3], index=['a','b','c'])
s2 = Series([4,5,6,7], index=['a','b','d','c'])
print(s1)
print(s2)
print(s1 + s2)      # 인덱스 명이 일치하는 요소끼리 더한다. 일치하는 인덱스가 없을 경우 NaN 반환
print(s1.add(s2))       # sub, mul, div 함수도 있다.

print()
df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), index=['서울', '대전', '부산'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), index=['서울', '대전', '제주', '수원'])
print(df1)
print(df2)
print(df1 + df2)        # 옵션 주는것 불가.
print(df1.add(df2))     
print(df1.add(df2, fill_value=0))   # add, sub, mul, div 함수로 연산하면 옵션을 줄 수 있다.     

print('-----'*5)
df = DataFrame([[1.4, np.nan],[7, 4.3], [np.NaN, np.NAN], [0.4, -2]], columns=['one', 'two'])
print(df)
print(df.isnull())      # null 있는지 없는지 bool 타입으로 보여줌
print(df.notnull())

print(df.drop(1))       # 원본에는 영향 주지 않음
print(df.dropna())      # NaN 있는 행을 날려버림
print(df.dropna(how='any'))     # NaN 하나라도 있을때
print(df.dropna(how='all'))     # 행의 모든 값이 NaN일때
print(df.dropna(subset=['one'])) # 특정 열에 NaN이 있을때 해당 행 drop
print(df.dropna(axis='rows'))   # 행 기준 NaN 포함 행 날림
print(df.dropna(axis='columns'))# 열 기준 NaN 포함 열 날림
print(df.fillna(0))     # NaN 값에 0으로 채움

print('기술적 통계 관련 함수')
print(df.sum())     # 열의 합계
print(df.sum(axis=0))# 열의 합계
print(df.sum(axis=1))# 행의 합계
print(df.mean(axis=0, skipna=True))     # 열기준 NaN은 스킵하여 평균 구하기
print()
print(df.describe())        # 여러 개의 통계 결과를 보여준다. 여기서 max 값을 벗어나면 outlier
print(df.info())        # 데이터 프레임의 구조를 설명해줌. 

