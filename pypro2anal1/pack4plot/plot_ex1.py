# 시각화 : 많은 양의 자료를 시각화를 통해 전체적인 데이터 분포, 패턴, 인사이트를 확인 가능함
# matplotlib 라이브러리를 이용 - seaborn이 추가적인 지원

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지

x = ['서울', '인천', '수원']      # x축, y축에는 숫자만 들어간다. 한글이 들어가는 이유는 리스트 형태에서 순번이 들어가기 때문에 입력 가능. 즉 인덱스 값으로 들어감
y = [5, 3, 7]
plt.xlim([-1,3])        # x축 최대값
plt.ylim([0,10])        # y축 최대값
plt.xlabel('지역')        # x축 라벨링
plt.ylabel('숫자')        # y축 라벨링
plt.title('제목')         # 차트의 이름
plt.yticks(list(range(0,10,3)))     # y축 칸 넓이 설정.
plt.plot(x, y)
plt.show()


data = np.arange(1,11,2)
plt.plot(data)
x = [0,1,2,3,4]
for a, b in zip(x, data):       # zip 함수는 데이터 길이가 동일한 리스트끼리 요소를 묶는 함수이다.
    plt.text(a, b, str(b))
plt.show()

x = np.arange(10)
y = np.sin(x)
print(x, y)
# plt.plot(x, y, 'bo')
# plt.plot(x, y, 'r+')
plt.plot(x, y, 'go-.', linewidth=2, markersize=12)  # g는 색상, o는 좌표 모양, linewidth는 선 두께, markersize는 좌표 크기
plt.show()

# 홀드 : 여러 개의 plot 명령을 겹쳐서 그리기
x = np.arange(0,3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.figure(figsize=(10,5))      # figsize 속성으로 그래프의 넓이를 넓혀준다.
plt.plot(x, y_sin,'r')  # 직선
plt.scatter(x, y_cos)   # 산점도
plt.xlabel('x축')
plt.ylabel('y축')
plt.legend(['sine & cosine'])
plt.show()

# subplot : 한 개의 그림 영역을 여러 개로 분리
plt.subplot(2, 1, 1)    # 2행 1열 모양으로 분리. 1번째 행에 삽입
plt.plot(x, y_sin)
plt.title('sine')
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('cosine')

fig = plt.gcf()
plt.show()
fig.savefig('test1.png')    # chart to image save

# 이미지 읽기
from matplotlib.pyplot import imread
img = imread('test1.png')
plt.imshow(img)
plt.show()
