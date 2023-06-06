# 공분산 / 상관계수
# 상관계수는 밀도를 숫자로 표현한다. 밀도를 가지고 상관관계를 정확하게 표현하기 힘들다. 
# 그래서 숫자화 해야 한다. 이 것을 정도에 따라 구분한 것 중 하나가 피어슨 상관계수다. 
# 이는 두 변수 간의 관련성을 알기 위해 이용된다.
# 상관계수 r은 -1 ~ 1 사이의 값을 갖는다. (±1 : 완전상관(밀도 촘촘), 0 : 상관관계 없다)
# 상관계수 r은 공분산을 표준화하면 얻을 수 있다.

import numpy as np
import matplotlib.pyplot as plt

# 공분산 : 데이터의 관계만 보여준다.(공분산의 한계)
print(np.cov(np.arange(1,6), np.arange(2,7)))   # 2.5
print(np.cov(np.arange(1,6), (3,3,3,3,3)))   # 0
print(np.cov(np.arange(1,6), np.arange(6,1,-1)))   # -2.5

print(np.cov(np.arange(1,6), np.arange(20,70,10)))   # 25
print(np.cov(np.arange(1,6), np.arange(2000,7000,1000)))   # 2500

# plt.scatter(np.arange(1,6), np.arange(2000,7000,1000))    
# plt.scatter(np.arange(1,6), (3,3,3,3,3))
# plt.scatter(np.arange(1,6), np.arange(6,1,-1))
# plt.show()

print('-------'*10)
x = [8,3,6,6,9,4,3,9,3,4]
print('x평균 : ', np.mean(x))
print('x분산 : ', np.var(x))

y = [6,2,4,6,9,5,1,8,4,5]
print('y평균 : ', np.mean(y))
print('y분산 : ', np.var(y))
plt.scatter(x,y)
plt.show()
print('x, y의 공분산 : ', np.cov(x,y))      # 5.2222
print('x, y의 공분산 : ', np.cov(x,y)[0,1])      # 5.2222. 양의 관계가 있다.

print('상관계수')
print('x, y의 상관계수 : ', np.corrcoef(x,y))      # 0.8663686
print('x, y의 상관계수 : ', np.corrcoef(x,y)[0,1])      # 0.8663686   양의 관계가 있다.
# y 값에 10ⁿ을 하더라도 상관계수는 동일하다. 공분산은 10ⁿ 만큼 증가.
# 상관계수는 1보다는 작고 0.3±0.05 보다는 커야 유의미하다.
# 주의사항 : 공분산 / 상관계수는 선형적인 데이터에 대해 유의하다. 비선형 데이터는 제대로 측정 불가.






