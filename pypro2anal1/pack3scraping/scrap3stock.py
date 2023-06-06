# 네이버 제공 코스피 정보 읽기 : csv파일로 저장 -> df
import requests
from bs4 import BeautifulSoup
import csv

url = 'https://finance.naver.com/sise/sise_market_sum.naver'
fname = '코스피.csv'
fObj = open(fname, mode='w', encoding='utf-8', newline='')
writer = csv.writer(fObj)

title = 'N    종목명    현재가    전일비    등락률    액면가    시가총액    상장주식수    외국인비율    거래량    PER    ROE'
title = title.split()
# print(title)
writer.writerow(title)

for page in range(1,3):
    data = requests.get(url.format(str(page)))
    data.raise_for_status()        # 읽기 실패하면 에러를 던지고 종료함
    soup = BeautifulSoup(data.text, 'html.parser')
    # print(soup)
    datas = soup.find('table', attrs={'class':'type_2'}).find("tbody").find_all("tr")
    # print(datas)
    for row in datas:
        cols = row.findAll('td')
        if len(cols) <= 1: continue     # [''] 처리
        # print(cols)
        da = [c.get_text().strip() for c in cols]
        # print(da)
        writer.writerow(da)
fObj.close()

import pandas as pd
import numpy as np

aa = pd.read_csv(fname)
df = pd.DataFrame(aa)
# re_df = df.apply(lambda x:x.str.strip(","), axis=1)
print(df.head(3))
print(df.tail(3))
# print(df['현재가'].std())

pri = df['현재가']
li =[]
for p in pri:
    li.append(int(p.replace(",",'')))
print(np.std(li))

        