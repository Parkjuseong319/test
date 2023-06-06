import urllib.request as req
from bs4 import BeautifulSoup

url = 'https://datalab.naver.com/'
webtoon = req.urlopen(url)

soup = BeautifulSoup(webtoon.read(), 'lxml')
re = soup.select("div.keyword_carousel > div > div > div:nth-child(10) > div > div > ul > li")
title = soup.select_one("div.keyword_carousel > div > div > div:nth-child(10) > div > strong > span")

print(title.text)
for i in re:
    text = i.find("em").text
    text2 = i.find('span').text
    print(text, text2)

# 강주가 한거

# url = urlopen("https://datalab.naver.com/")
# soup = BeautifulSoup(url.read(),'html.parser')
#
# target_date = '2023.04.22.(토)'
# rankings2 = soup.select('div.rank_inner')
#
# for rank in rankings2:
#     if rank.find('span',class_='title_cell').string == target_date:
#
#         print(target_date,"자료")
#
#         for i,ra in enumerate(rank.findAll('span',class_='title')):
#             print(int(i+1),"위 -",ra.text)

