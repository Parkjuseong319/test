# 상관관계 분석
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False 

data = pd.read_csv('../testdata/drinking_water.csv')
print(data.head(3))

print('표준편차 출력')
print(np.std(data.친밀도)) # 0.9685
print(np.std(data.적절성)) # 0.8580
print(np.std(data.만족도)) # 0.8271

print('공분산 출력')
print(np.cov(data.친밀도, data.적절성)) # numpy function. 0.4164
print(np.cov(data.친밀도, data.만족도)) # 0.3756
print()
print(data.cov())       # pandas function
"""
          친밀도       적절성       만족도
친밀도  0.941569  0.416422  0.375663
적절성  0.416422  0.739011  0.546333
만족도  0.375663  0.546333  0.686816
"""

print('상관계수 출력')
print(np.corrcoef(data.친밀도, data.적절성)) # 0.3756
print(np.corrcoef(data.친밀도, data.만족도)) # 0.3756
print()
print(data.corr())      # pandas function
print(data.corr(method='pearson'))      # 변수가 등간, 비율 척도일 때
print(data.corr(method='spearman'))     # 변수가 서열 척도일 때
print(data.corr(method='kendall'))      # 변수가 서열 척도일 때

"""
일반적으로
r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
로 해석한다.
"""

co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))    # 만족도에 대하여 내림차순으로 볼 수 있다.

print()
# data.plot(kind = 'scatter', x='만족도', y='적절성')
# plt.show()

import seaborn as sns
# sns.heatmap(data.corr())
# plt.show()

# heatmap에 텍스트 표시 추가사항 적용해 보기
corr = data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)  # 상관계수값 표시
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask and correct aspect ratio
vmax = np.abs(corr.values[~mask]).max()
fig, ax = plt.subplots()     # Set up the matplotlib figure

sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax, square=True, linecolor="lightgray", linewidths=1, ax=ax)

for i in range(len(corr)):
    ax.text(i + 0.5, len(corr) - (i + 0.5), corr.columns[i], ha="center", va="center", rotation=45)
    for j in range(i + 1, len(corr)):
        s = "{:.3f}".format(corr.values[i, j])
        ax.text(j + 0.5, len(corr) - (i + 0.5), s, ha="center", va="center")
ax.axis("off")
plt.show()
