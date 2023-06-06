# 특정 신문사 사이트에서 검색 단어에 의한 검색결과를 읽은 후 명사를 추출해 워드 클라우드 그리기
# pip install pytagcloud
# pip install simplejson

import urllib.request
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pytagcloud
from urllib.parse import quote      # 인코딩해주는 모듈

# keyword = input('검색어:')
keyword = '축제'
print(keyword)
print(quote(keyword))
url = 'https://www.donga.com/news/search?query=' + quote(keyword)   # 웹상에서 검색 처리할때 꼭 quote 모듈 사용해서 인코딩 해야한다.
print(url)
source_code = urllib.request.urlopen(url)

soup = BeautifulSoup(source_code, 'lxml', from_encoding='utf-8')
# print(soup)
msg = ""
for title in soup.select('div.rightList > span.tit'):
    title_link = title.find('a')
    # print(title_link)
    article_url = title_link['href']
    # print(article_url)
    try:
        source_article = urllib.request.urlopen(article_url)    # 실제 기사 읽기
        soup = BeautifulSoup(source_article, 'lxml', from_encoding='utf-8')
        content = soup.select("div.article_txt")
        # print(content)
        for imsi in content:
            item = str(imsi.find_all(text=True))
            # print(item)
            msg = msg + item    # 각 tag의 text를 누적
    except:
        pass
# print(msg)

from collections import Counter     # 단어 수를 세어주는 라이브러리

okt = Okt()
nouns = okt.nouns(msg)  # 명사별로 분할

result = []
for imsi in nouns:
    if len(imsi) > 1:
        result.append(imsi)
print(result)
print(result[:10])
count = Counter(result)
print(count)

# 워드 클라우드 작성
tag = count.most_common(50)     # 제일 빈도수 높은거 상위 n개 잡아준다.
taglist = pytagcloud.make_tags(tag, maxsize=100)
print(taglist[:10])     # [{'color': (123, 160, 25), 'size': 115, 'tag': '축제'},.....]

# image 생성 후 저장
pytagcloud.create_tag_image(taglist, 'word.png', size=(1000,650), background=(0,0,0), fontname='korean', rectangular=False)
# 한글이 깨질땐 C:\Users\Park\AppData\Roaming\Python\Python310\site-packages\pytagcloud\fonts 로 들어가서 폴더내에 한글 지원 폰트를 넣고
# font.json 에 들어가서 추가해줘야한다. 거기서 이름을 자유롭게 바꿔서 넣어주고 그 이름 호출하여 사용하면 된다.

# image read
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()

# 브라우저로 읽기
import webbrowser
webbrowser.open('word.png')






