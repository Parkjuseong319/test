# DataFrame 재구조화, 구간 기준값 설정, merge, group 처리
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(6).reshape(2,3) + 1000, index=['대전', '서울'], columns=['2021', '2022', '2023'])
print(df)
print(df.T)

print()
df_row = df.stack()     # index를 기준으로 column 쌓기
print(df_row)

df_col = df_row.unstack()   # stack 원복
print(df_col)

print('범주화------------------')
price = [10.3, 5.5, 7.8, 3.6]   # 기준값과 비교할 대상
cut = [3, 7, 9, 11]     # 구간 기준값
result_cut = pd.cut(price, cut)     # 연속형 자료를 범주형으로 구간 설정해준다.
print(result_cut)   # (9, 11] : 9 < x <= 11, (3, 7] : 3 < x <= 7, (7, 9] : 7 < x <= 9, (3, 7] : 3 < x <=7
print(pd.value_counts(result_cut))
print()
datas = pd.Series(np.arange(1, 1001))
print(datas.head(2))
print(datas.tail(2))
result_cut2 = pd.qcut(datas, q=3)       # q는 나눌 구간의 개수이다. Categories : 3
print(result_cut2)
print(pd.value_counts(result_cut2))

# 각 범주 값을 그룹별 처리
group_col = datas.groupby(result_cut2)
print(group_col.agg(['count', 'mean', 'std', 'min']))       # agg는 함수 이름만 불러줘도 함수 실행해준다.

print()
def mySummaryFunc(gr):
    return {
            'count':gr.count(),
            'mean':gr.mean(),
            'std':gr.std(),
            'min':gr.min(),
        }

print(group_col.apply(mySummaryFunc))   
print(group_col.apply(mySummaryFunc).unstack())     
# apply를 통해 함수 실행 시키면 stack처럼 컬럼 기준으로 값이 쌓이는데 그걸 unstack 함수를 써주면 agg() 함수로 호출한 거처럼 된다.

# merge
print('merge : 데이터프레임 병합')
df1 = pd.DataFrame({'data1':range(7), 'key':['b', 'b', 'a', 'c', 'a', 'a', 'b']})
print(df1)
df2 = pd.DataFrame({'key':['a', 'b', 'd'],'data2':range(3)})
print(df2)

print()
print(pd.merge(df1,df2, on='key'))      # inner join
print(pd.merge(df1,df2, on='key', how='inner'))      # inner join
print(pd.merge(df1,df2, on='key', how='outer'))      # full outer join, 일치 하지 않는 것은 NaN이다.
print(pd.merge(df1,df2, on='key', how='left'))       # left outer join, 일치 하지 않는 것은 NaN이다.
print(pd.merge(df1,df2, on='key', how='right'))      # right outer join, 일치 하지 않는 것은 NaN이다.

print()
df3 = pd.DataFrame({'key2':['a', 'b', 'd'],'data2':range(3)})
print(df3)
print(pd.merge(df1,df3, left_on='key', right_on='key2'))        # 공통 칼럼이 없을 때 이렇게 join 가능하다.

print()
print(pd.concat([df1, df3]))        # dataframe 이어 붙이기. 기본적으로 행단위로 붙임
print(pd.concat([df1, df3], axis=0))# 행단위로 붙이기
print(pd.concat([df1, df3], axis=1))# 열단위로 붙이기

print('그룹 단위의 처리 : group by,pivot')
data = {
        'city':['강남', '강북', '강남', '강북'],
        'year':[2000, 2001, 2002, 2002],
        'pop':[3.3, 2.5, 3.0, 2.0],
    }
df = pd.DataFrame(data)
print(df)
print()
# DataFrame 구조 재구성
print(df.pivot(index= 'year', columns='city', values='pop'))    # index, columns에는 범위형 데이터가, values에는 연속형 데이터를 줘야함
print()
print(df.pivot(index= 'city', columns='year', values='pop'))
print()
print(df['pop'].describe())

print()
# DataFrame 자료 그룹화
hap = df.groupby(by=['city'])   # SQL group by와 같다
print(hap.sum())

print(df.groupby(by=['city']).sum())
print(df.groupby(by=['city']).mean())

print('\npivot_table = pivot + groupby')
print(df)
print()
print(df.pivot_table(index=['city']))   # pivot_table의 기본 함수는 mean()이 적용
print(df.pivot_table(index=['city'], aggfunc=np.mean))
print(df.pivot_table(index=['city', 'year'], aggfunc=np.mean))  # stack의 형태로 나온다.
print(df.pivot_table(index=['city', 'year'], aggfunc=[len, np.sum]))
print()
print(df.pivot_table(index='city', values=['pop']))
print(df.pivot_table(index='city', values=['pop'], aggfunc=np.mean))
print()
print(df.pivot_table(values=['pop'], index=['year'], columns=['city']))
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], fill_value=0))
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], fill_value=0, margins=True))     
# margins 속성이 True일때 aggfunc의 함수를 따라 각 행,열의 대한 값을 구한다.
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], fill_value=0, margins=True, aggfunc=np.sum))

