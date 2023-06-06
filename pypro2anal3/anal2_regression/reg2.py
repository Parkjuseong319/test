# 단순선형회귀 분석 모델 작성 방법4
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IQ와 시험성적 간의 인과관계를 확인하고 시험 점수 값 예측하기
score_iq = pd.read_csv('../testdata/score_iq.csv')
print(score_iq.head(3))
print(score_iq.info())

x = score_iq.iq
y = score_iq.score

# 상관계수
print(np.corrcoef(x, y))    # 0.88222034
print(score_iq.corr())

# plt.scatter(x, y)
# plt.show()

# 인과관계는?    선형회귀분석을 통해 알 수 있다.
model = stats.linregress(x, y)
print(model)
# LinregressResult(slope=0.6514309527270081, intercept=-2.856447122197551, rvalue=0.8822203446134705,
# pvalue=2.8476895206672287e-50, stderr=0.028577934409305377, intercept_stderr=3.54621191804853)
# p-value < 0.05 일때 인과관계가 있고 이 모델은 유의하다라고 해석 할 수 있다. 
print('slope : ', model.slope)
print('intercept : ', model.intercept)
print('p-value : ', model.pvalue)
print('stderr(표준차) : ', model.stderr)

plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept)
plt.show()

# 점수예측
print('점수 예측 : ', model.slope * 140 + model.intercept)

# predict() 제공 X. 대신의 numpy의 polyval()을 사용할 수 있다.
print('예측 값 : ', np.polyval([model.slope, model.intercept], np.array(score_iq['iq'][:5])))

print("실제 값 : ", score_iq['score'][:5].values)

# 새로운 iq 값으로 성적예측
new_df = pd.DataFrame({'iq':[55,66,77,88,99,123]})
print('예측 값 : ', np.polyval([model.slope, model.intercept], new_df))

