# json 문서 읽기
import json
import urllib.request as req
import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json'
plainText = req.urlopen(url).read().decode()
# print(plainText, type(plainText))

jsonData = json.loads(plainText)            # load는 str타입을 dict로 convert함.
# print(jsonData, type(jsonData))

# print(jsonData.get('SeoulLibraryTime').get('row'))
libData = jsonData.get('SeoulLibraryTime').get('row')

print()
datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    # print(name, ' ', tel, ' ', addr)
    imsi = [name, tel, addr]
    datas.append(imsi)
    
# print(datas)
df = pd.DataFrame(datas, columns=['이름', '전화', '주소'])
print(df)

