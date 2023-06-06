# 서울지역 cctv 설치 현황 + 인구수 데이터로 시각화
# pip install xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

cctv_seoul = pd.read_csv('../testdata/cctv_seoul.csv')
cctv_seoul.rename(columns={cctv_seoul.columns[0]:'구별'}, inplace=True)
del cctv_seoul['2013년도 이전']     # 불필요한 열 삭제
del cctv_seoul['2014년']
del cctv_seoul['2015년']
del cctv_seoul['2016년']
print(cctv_seoul.head())
print(cctv_seoul.shape)     # (25, 2)

pop_seoul = pd.read_excel('../testdata/population_seoul.xls', sheet_name='YainSoft_Excel1', header=2 ,skiprows=[3], usecols='B,D,G,J,N' )
pop_seoul.rename(columns={pop_seoul.columns[0]:'구별',
                          pop_seoul.columns[1]:'인구수',
                          pop_seoul.columns[2]:'한국인',
                          pop_seoul.columns[3]:'외국인',
                          pop_seoul.columns[4]:'고령자'}, inplace=True)
print(pop_seoul.head())
print(len(pop_seoul['구별'].unique()))

pop_seoul['외국인비율'] = pop_seoul['외국인']/pop_seoul['인구수'] * 100
pop_seoul['고령자비율'] = pop_seoul['고령자']/pop_seoul['인구수'] * 100
print(pop_seoul.head())

# merge : cctv, population
data_result = pd.merge(cctv_seoul, pop_seoul, on='구별')
print(data_result.head())
print(data_result.shape)    # (25, 8)

data_result.set_index('구별', inplace=True)
print(data_result.head())

plt.figure()
data_result['소계'].plot(kind='barh', grid=True, figsize=(10,10))
plt.show()

data_result['cctv비율'] = data_result['소계'] / data_result['인구수'] * 100
data_result['cctv비율'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
plt.show()

plt.figure(figsize=(7,7))
plt.scatter(data_result['인구수'], data_result['소계'], s=50)
plt.xlabel('인구수')
plt.ylabel('cctv설치 수')
plt.grid()
plt.show()






