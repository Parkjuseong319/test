# Markup Languege로 구성된 웹 문서 읽기 - beautifulsoup
from bs4 import BeautifulSoup

html_page = """
<html>
<body>
<h1>제목 태그</h1>
<p>아름다운 금요일</p>
<p>화창한 금요일</p>
</body>
</html>
"""
print(type(html_page))

soup = BeautifulSoup(html_page, "html.parser")      # Beautifulsoup 객체가 생성
print(type(soup))
print(soup)
print()
h1 = soup.html.body.h1
print(h1.string, "  ", h1.text)

p1 = soup.html.body.p
print(p1.text)

p2 = p1.next_sibling.next_sibling       # 닫는 태그까지 걸리기 때문에 next_sibling 두번 사용해서 다음 태그로 넘어같다.
print(p2.text)

print('\nfind(반환값이 한개) 메소드 사용')
html_page2 = """
<html>
<body>
<h1 id='title'>제목 태그</h1>
<p>아름다운 금요일</p>
<p id="my" class="our">화창한 금요일</p>
</body>
</html>
"""
soup2 = BeautifulSoup(html_page2, "lxml")
# element, attribute 둘 중 하나로 찾을 수 있다.
print(soup2.p, '  ', soup2.p.string)
print(soup2.find('p').string)       # find 함수는 최초의 태그 하나만.
print(soup2.find('p', id='my').text)
print(soup2.find(id='my').text)
print(soup2.find(id='title').text)
print(soup2.find(class_='our').string)  # class_ 로 클래스로 찾을 수 있다.
print(soup2.find(attrs={'class':'our'}).string)     # attrs 로  찾기 가능하다.
print(soup2.find(attrs={'id':'my'}).string)

print('\nfind_all/findAll(반환값이 여러개) 메소드 사용')
html_page3 = """
<html>
<body>
<h1 id='title'>제목 태그</h1>
<p>아름다운 금요일</p>
<p id="my" class="our">화창한 금요일</p>
<div>
    <a href="https://www.naver.com">naver</a>
    <a href="https://www.daum.net">daum</a>
</div>
</body>
</html>
"""
soup3 = BeautifulSoup(html_page3, "lxml")
print(soup3.find_all(['a']))
print(soup3.find_all('p'))
print(soup3.find_all(['a','p']))        # 태그 여러개를 지정시 리스트로 묶는다.
print(soup3.findAll('p'))
print()
links = soup3.find_all('a')
for i in links:
    href = i.attrs['href']
    text = i.string
    print(href, "--->", text)

print('정규 표현식')
import re
links2 = soup3.find_all(href=re.compile(r'^https'))
print(links2)
for j in links2:
    print(j.attrs['href'])
    
# print('bugs 음악 차트 title 얻기')
# from urllib.request import urlopen
# url = urlopen("https://music.bugs.co.kr/chart")
# soup = BeautifulSoup(url.read(), 'html.parser')
# # print(soup)
# musics = soup.find_all('td', class_="check")
# # print(musics)
# for i, music in enumerate(musics):
#     print("{}위:{}".format(i+1,music.input['title']))
    
print('\nselect : css의 selector를 사용')
html_page4 = """
<html>
<body>
<div id="hello"> 
    <a href="https://www.daum.net">daum</a>
    <span>
        <a href="https://www.naver.com">naver</a>
    </span>
    <ul class="world">
        <li>안녕</li>
        <li>반가워</li>
    </ul>
    <ul class="world2">
        <li>안녕못해</li>
        <li>반갑지 않아</li>
    </ul>
</div>
<div id='hi' class='good'>
    second div
</div>
</body>
</html>
"""
soup4 = BeautifulSoup(html_page4, "lxml")
aa = soup4.select_one("div#hello > a")      # 단수 선택. 부등호는 직계 태그 지정할때 사용  
print(aa,' ', aa.string)
bb = soup4.select("div#hello ul.world > li")     # 복수 선택
print(bb)
for i in bb:
    print("li : ", i.string)

print()
import pandas as pd

msg = list()
for i in bb:
    msg.append(i.string)

df = pd.DataFrame(msg, columns=['메세지'])
print(df)






