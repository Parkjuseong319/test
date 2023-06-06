# 데이터의 성격, 크기에 따라 차트의 종류를 효율적으로 선택해야한다.
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# 차트 표현 방법 (스타일 인터페이스)
x = np.arange(10)

# 1) matplot style의 interface
plt.figure()
plt.subplot(2, 1, 1)    # row, column, panel number
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()

# 2) 객체 지향 인터페이스
fig, ax = plt.subplots(nrows=2, ncols=1)    # fig는 전체를 받고 ax는 낱개로 받는다.
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()

# 1번 방법과 유사
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.hist(np.random.randn(10), bins=5, alpha=0.7)       # bins는 구간, alpha는 선명도. 최대가 1 최저 0
ax2.plot(np.random.randn(10))
plt.show()

# 막대 그래프(차트)
data = [50,80, 100, 55, 90]     # 데이터 양이 적고 이산 데이터일때 막대 그래프 사용
plt.bar(range(len(data)), data) # 세로막대
plt.show()

plt.barh(range(len(data)), data)# 가로막대
plt.show()

# 원 그래프(pi chart)
plt.pie(data, explode=(0, 0.2, 0, 0, 0), colors=['yellow', 'red', 'blue'])  # explode 속성으로 조각끼리 사이를 떨어트릴 수 있다.
plt.show()

# 박스 그래프(box plot)
plt.boxplot(data)
plt.show()

# pandas의 시각화
import pandas as pd
fdata = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2000', periods=1000), columns=list('abcd'))
fdata = fdata.cumsum()
print(fdata.head(3))
print(fdata.tail(3))
plt.plot(fdata)
plt.show()

fdata.plot()        # pandas의 시각화 기능
fdata.plot(kind='bar')
fdata.plot(kind='box')
plt.xlabel('time')
plt.ylabel('data')
plt.show()




