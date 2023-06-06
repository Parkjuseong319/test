# xml 문서 읽기
from bs4 import BeautifulSoup

with open("../testdata/my.xml", mode='r', encoding='utf-8') as f:
    xmlfile = f.read()
    print(xmlfile, type(xmlfile))   # xml파일 읽은 것은 str 타입
    
soup = BeautifulSoup(xmlfile, 'lxml')
print(type(soup))
itemTag = soup.findAll('item')
print(itemTag[0])

print()
for i in itemTag:
    nameTag = i.findAll('name')
    for j in nameTag:
        print('id : ' + j['id'] + " name : " + j.string)
    print('tel : ' + i.find('tel').string )
    for j in i.findAll('exam'):
        print('kor : ' + j.attrs['kor'] + " eng : " + j.attrs['eng'])
    
