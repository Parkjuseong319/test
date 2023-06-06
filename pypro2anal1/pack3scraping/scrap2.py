# web scraping test
import urllib.request as req
import urllib
from bs4 import BeautifulSoup

# 위키백과에서 임의의 검색된 결과 읽기
url = 'https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0'
wiki = req.urlopen(url)
# print(wiki.read())
soup = BeautifulSoup(wiki.read(), 'lxml')
# print(soup)
#mw-content-text > div.mw-parser-output > p:nth-child(6)
result = soup.select("div.mw-parser-output > p")
# print(result)
for ss in result:
    for s in ss:
        if s.string != None:
            print(s.string)
print("-----"* 5)
url = 'https://news.daum.net/society#1'
daum = req.urlopen(url)
soup2 = BeautifulSoup(daum.read(), 'lxml')
data = soup2.select_one('div.direct-link > a')
print(data)

datas = soup2.select('div.direct-link > a')
print(datas)
for i in datas:
    href = i.attrs['href']
    text = i.text
    print('href:%s, text:%s'%(href, text))


print()
datas2 = soup.find_all('a')
# print(datas2)
for i in datas2[:5]:
    href = i.attrs['href']
    text = i.text
    print('href:%s, text:%s'%(href, text))

print("----- naver : 자주 변경되는 자료 (주식, 환율 등) -------------------")
import datetime
import time

def working():
    url = 'https://finance.naver.com/marketindex/'
    data = req.urlopen(url)
    soup = BeautifulSoup(data, 'lxml')
    price = soup.select_one("div.head_info > span.value").string
    print('미국 USD : ', price)
    t = datetime.datetime.now()
    fname = "./test/" + t.strftime('%Y-%m-%d-%H-%M-%S') + ".txt"
    
    with open(fname, mode='w') as f:
        f.write(price)
    
    
while True:
    working()
    time.sleep(3)
    


