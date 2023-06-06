# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

from io import StringIO

# 원래 수업에서는 파일로 만들어서 사용하였다.
data = StringIO(        
"""
요일,외식유무,소득수준
토,0,57
토,0,39
토,0,28
화,1,60
토,0,31
월,1,42
토,1,54
토,1,65
토,0,45
토,0,37
토,1,98
토,1,60
토,0,41
토,1,52
일,1,75
월,1,45
화,0,46
수,0,39
목,1,70
금,1,44
토,1,74
토,1,65
토,0,46
토,0,39
일,1,60
토,1,44
일,0,30
토,0,34
""")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score

fdata = pd.read_csv(data)
print(fdata.head(2), fdata.shape)   # (28, 3)
mydata = fdata.loc[(fdata['요일']=='토') | (fdata['요일']=='일')]
print(mydata.head(2), mydata.shape) # (21, 3)   평일 데이터 삭제 후

# 가설 설정 가능
# 귀무 : 소득수준과 외식유무는 인과관계가 없다.
# 대립 : 소득수준과 외식유무는 인과관계가 있다.

model = smf.glm(formula='외식유무 ~ 소득수준', data = mydata, family=sm.families.Binomial()).fit()   
print(model.summary())
print()
pred = model.predict(mydata)
print('예측값 : ', np.around(pred[:5].values))     # [1. 0. 0. 0. 1.]
print('실제값 : ', mydata['외식유무'][:5].values)      # [0 0 0 0 1]
print('분류 정확도 : ', accuracy_score(mydata['외식유무'], np.around(pred))) # 0.90476190
print()
new_input_data = pd.DataFrame({'소득수준':[int(input("소득수준을 입력해봐 : "))]})
# print(np.rint(model.predict(new_input_data)))
print('외식함' if np.rint(model.predict(new_input_data))[0] == 1 else '외식 못함')



