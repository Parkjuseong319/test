# 이원카이제곱
# 동질성 검정 - 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다.
#  두 집단 이상에서 각 범주(집단) 간의 비율이 서로 동일한가를 검정하게 된다.
#  두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다.

# 동질성 검정실습1) 교육방법에 따른 교육생들의 만족도 분석 - 동질성 검정 survey_method.csv
# 귀무 : 교육방법에 따른 교육생들의 만족도에 없다.
# 대립 : 교육방법에 따른 교육생들의 만족도에 있다.

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('../testdata/survey_method.csv')
print(data.head(3))
print(data.info())
print(data['method'].unique())  # list type. [1,2,3]
print(set(data['survey']))      # {1, 2, 3, 4, 5}

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족','만족','보통','불만족', '매우불만족']
ctab.index = ['방법1', '방법2', '방법3']
print(ctab)

# chi_result = [ctab.loc['방법1'],ctab.loc['방법2'],ctab.loc['방법3']]
# chi2, p, df, _ = stats.chi2_contingency(chi_result)

chi2, p, df, _ = stats.chi2_contingency(ctab)
print('chi2 :{}, p :{}, df :{}'.format(chi2, p, df))
# chi2 :6.544667820529891, p :0.5864574374550608, df :8, cv 15.51
# 해석 : p-value : 0.5865 > 0.05 이므로 귀무채택
# 위 데이터(조산된 관찰값)는 우연히 발생한 것이다.

print('----' * 15)
# 동질성 검정 실습2) 연령대별 sns 이용률의 동질성 검정
# 20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용 현황을 조사한 자료를 바탕으로 연령대별로 홍보
# 전략을 세우고자 한다.
# 연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자

# 귀무 : 연령대별로 sns 서비스 이용률이 서로 동일하다. (동질이다)
# 대립 : 연령대별로 sns 서비스 이용률이 서로 다르다.   (동질이 아니다)
data2 = pd.read_csv('../testdata/snsbyage.csv')
print(data2.head(3))
print(data2.info())
print(data2['age'].unique())            # [1 2 3]. 20대, 30대, 40대
print(data2['service'].unique())        # ['F' 'T' 'K' 'C' 'E']

ctab2 = pd.crosstab(index=data2['age'], columns=data2['service'])
print(ctab2)

chi2, p, df, _ = stats.chi2_contingency(ctab2)
print('chi2 :{}, p :{}, df :{}'.format(chi2, p, df))
# chi2 :102.75202494484225, p :1.1679064204212775e-18, df :8
# 결론 : p-value : 1.1679064204212775e-18 < 0.05 이므로 귀무 기각
# ∴ 연령대별로 sns 서비스 이용률이 서로 다르다.
