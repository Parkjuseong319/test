# Two-way ANOVA(이원 배치 분산분석) : 목적 요인이 두 개인 경우
# 태아수와 관측자수 요인에 따른 집단이 태아의 머리둘레 평균값에 영향을 주는가 검증
# 귀무 : 태아수와 관측자수는 태아의 머리둘리 평균값에 영향을 주지않는다.
# 대립 : 태아수와 관측자수는 태아의 머리둘리 평균값에 영향을 준다.
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data = pd.read_csv("../testdata/group3_2.txt")
print(data.head(3),data.shape)
print(set(data['태아수']),set(data['관측자수']))# {1, 2, 3} {1, 2, 3, 4}

# data.boxplot(column='머리둘레',by='태아수')
# plt.show()
# data.boxplot(column='머리둘레',by='관측자수')
# plt.show()

#교호작용 혹은 상호작용 (interaction)효과 : 특정변인의 효과가 다른 변인의 수준에 따라 달라지는 것.
#ANOVA 분석 시 상호작용을 뺀 경우
reg = ols("data['머리둘레']~C(data['태아수'])+C(data['관측자수'])",data=data).fit()#머리둘레 종속변수 태아수,관측자수는 독립변수가된다.
#선형회귀무터 모델이 만들어진다. 카이제곱, 카이스퀘어,티테스트,등등 모델은 만들어진다고 이야기한적없다.
#여기서 부터 fit이 등장한다.
result = anova_lm(reg,type=2)
print(result)
"""
                   df      sum_sq     mean_sq            F        PR(>F)
C(data['태아수'])    2.0  324.008889  162.004444  2023.182239  1.006291e-32
C(data['관측자수'])   3.0    1.198611    0.399537     4.989593  6.316641e-03
Residual         30.0    2.402222    0.080074          NaN           NaN
"""
print('-----------------------------')
#ANOVA 분석 시 상호작용을 적용한경우
reg = ols("머리둘레~C(태아수)+C(관측자수)+C(태아수):C(관측자수)",data=data).fit()
# 상호작용을 적용한 경우에는 C(태아수):C(관측자수) 이렇게 써줘야 적용된다.
result = anova_lm(reg,type=2)
print(result)
"""
                  df      sum_sq     mean_sq            F        PR(>F)
C(태아수)           2.0  324.008889  162.004444  2113.101449  1.051039e-27
C(관측자수)          3.0    1.198611    0.399537     5.211353  6.497055e-03
C(태아수):C(관측자수)   6.0    0.562222    0.093704     1.222222  3.295509e-01
Residual        24.0    1.840000    0.076667          NaN           NaN
"""

# pvalue(0.329 > 0.05) 이므로 귀무가설 채택이다.
 























