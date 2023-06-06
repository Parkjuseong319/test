# 표준편차, 분산의 중요성 - 평균 값은 같으나 표준편차, 분산값이 다르면 데이터의 분포가 달라진다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(1)
print(stats.norm(loc=1, scale=2).rvs(10))

centers = [1, 1.5, 2]
col = 'rgb'

# 표준편차값을 지정
std = 0.01
datas = []

for i in range(3):
    datas.append(stats.norm(loc=centers[i], scale=std).rvs(100))
    print(datas)
    plt.plot(np.arange(100) + i*100, datas[i], '*', color=col[i])
plt.show()

