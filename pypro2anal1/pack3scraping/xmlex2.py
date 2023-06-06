# 서울시가 제공하던 '강남 소재 도서관 정보(5행) XML 문서 읽기
import urllib.request as req
from bs4 import BeautifulSoup

url = 'https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.xml'
plainText = req.urlopen(url).read().decode()
# print(plainText)

xmlObj = BeautifulSoup(plainText, 'lxml')
libData = xmlObj.select("row")
# print(libData)
for data in libData:
    name = data.find('lbrry_name').text     # xml 파일을 켜서 볼땐 대문자로 보이지만 부를때 소문자로 써줘야 불러진다.
    addr = data.find('adres').text
    print('도서관 명 : ', name)
    print('주소 : ', addr, "\n")

