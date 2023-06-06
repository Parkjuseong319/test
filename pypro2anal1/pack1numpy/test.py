# data = np.array([[1,2,3,4],
#                 [5,6,7,8],
#                 [9,10,11,12],
#                 [13,14,15,16]])
#
# print(data)
# print(data[::-1, ::-1])
import numpy as np
import pandas as pd

# ran = np.random.randn(36).reshape(9,4)
df = pd.DataFrame(np.random.randn(36).reshape(9,4))
df.columns = ['가격1', '가격2', '가격3', '가격4']
print(df.mean(axis=0))
print(df)

# df = pd.DataFrame(np.arange(12).reshape((4,3)), index=['1월', '2월', '3월', '4월'], columns=['강남', '강북', '서초'])
# print(df)

from bs4 import BeautifulSoup
import urllib.request
url = "http://www.kyochon.com/menu/chicken.asp"

chicken = urllib.request.urlopen(url)
soup = BeautifulSoup(chicken.read(), 'lxml')
result = soup.select('#tabCont01 > ul > li')
menu = []
price = []

for re in result:
    name = re.find('dt').text 
    money = re.find('strong').text
    menu.append(name)
    price.append(money)

re_price =[]
for p in price:
    re_price.append(int(p.replace(",",'')))

df = pd.DataFrame(menu, columns=['상품명'])
df['가격'] = re_price
print('출력결과 : \n가격 평균 :',df['가격'].mean())
print('가격 표준편차 :',df['가격'].std())

    






