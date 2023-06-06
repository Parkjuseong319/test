# 기상청 제공 날씨 정보 XML 문서 읽기
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd

url = 'http://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108'
data = req.urlopen(url).read().decode()
# print(data)

soup = BeautifulSoup(data, 'lxml')
# print(soup)
title = soup.find('title').string
print(title)

city = soup.find_all('city')
# print(city)
cityDatas = []
for c in city:
    cityDatas.append(c.string)
# print(cityDatas)
df = pd.DataFrame()
df['city'] = cityDatas
# print(df.head(3))
# 부모 > 자식   부모    자손     노드1+노드2 (형제노드. 방향은 아래), 노드1-노드2(형제노드. 방향은 위
tempMins = soup.select('location > province + city + data > tmn')   # +는 아래 형제, -는 위에 형제(sibling)

tempDatas = []
for t in tempMins:
    tempDatas.append(t.string)
df['temp_min'] = tempDatas
df.columns = ['지역', '최저기온']
print(df.head(3))

# csv로 저장
df.to_csv('날씨정보.csv', index=False)
df2 = pd.read_csv('날씨정보.csv')
print(df2.head(3))

print(df.iloc[0:3, :])
print(df.iloc[0:3, 0:1])

print(df.loc[:, '지역'].head(3))

print()
print(df.info())
df = df.astype({'최저기온':int})
print(df.info())
print(df['최저기온'].mean())
print(df['최저기온'].std())

print(df.loc[df['최저기온'] >= 13])
