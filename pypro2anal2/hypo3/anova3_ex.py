import pandas as pd
import scipy.stats as stats

# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
# 귀무 : 기름의 종류에 따라 빵에 흡수된 기름의 양에 차이가 없다.
# 대립 : 기름의 종류에 따라 빵에 흡수된 기름의 양에 차이가 있다.
data = pd.read_csv('anova_oil.txt', sep=' ')
data.columns = ['종류', '양']
print(data.head(3), len(data))
print(data.isnull().sum())
data = data.fillna(data.mean())    # 평균으로 NaN값 대체
print(data.isnull().sum())
oil1 = data[data['종류'] == 1]
oil2 = data[data['종류'] == 2]
oil3 = data[data['종류'] == 3]
oil4 = data[data['종류'] == 4]

print(stats.levene(oil1['양'], oil2['양'], oil3['양'], oil4['양']).pvalue)  # pvalue=0.3268 등분산성 O
print(stats.shapiro(oil1['양']).pvalue)   
print(stats.shapiro(oil2['양']).pvalue)   
print(stats.shapiro(oil3['양']).pvalue)   
print(stats.shapiro(oil4['양']).pvalue)   # 정규성 O

f_sta, p_val = stats.f_oneway(oil1['양'], oil2['양'], oil3['양'], oil4['양'])
print("p-value : {}".format(p_val))    # p(0.8482) > 0.05이므로 귀무가설 채택 

