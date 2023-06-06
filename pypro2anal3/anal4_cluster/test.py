import requests

url = 'http://apis.data.go.kr/1270000/lawedu/lawqna'
params ={'serviceKey' : '서비스키', 'pageNo' : '1', 'numOfRows' : '10' }

response = requests.get(url, params=params)
print(response.content)