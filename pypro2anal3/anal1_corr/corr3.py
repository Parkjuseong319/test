# data.go.kr에 있는 자료를 사용
# 국내 유료관광지에 대해 외국인(일본, 중국, 미국) 관광객이 선호하는 관광지 관계분석
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
import json

# 그래프 작성 함수
def setScatterChart(tour_table, all_table, tourpoint):
    tour = tour_table[tour_table['resNm']==tourpoint]
    # print(tour)
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    # print(merge_table)
    
    fig = plt.figure()
    fig.suptitle(tourpoint + '상관관계 분석')
    
    plt.subplot(1,3,1)
    plt.xlabel('중국인')
    plt.ylabel('외국인 관광객 수')
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    # print('r1 : ', r1)
    plt.title('r={:.5f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], s=6, c='red')
    
    plt.subplot(1,3,2)
    plt.xlabel('일본인')
    plt.ylabel('외국인 관광객 수')
    lamb2 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    # print('r2 : ', r2)
    plt.title('r={:.5f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], s=6, c='blue')
    
    plt.subplot(1,3,3)
    plt.xlabel('미국인')
    plt.ylabel('외국인 관광객 수')
    lamb3 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lamb3(merge_table)
    # print('r3 : ', r3)
    plt.title('r={:.5f}'.format(r3))
    plt.scatter(merge_table['usa'], merge_table['ForNum'], s=6, c='green')
    plt.tight_layout()
    # plt.show()
    return [tourpoint, r1, r2, r3]

# 메인 함수
def start():
    fname = "../testdata/서울특별시_관광지입장정보_2011_2016.json"
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    # tour_table = pd.DataFrame(jsonTP)
    # print(tour_table.columns)   # ['ForNum', 'NatNum', 'addrCd', 'gungu', 'resNm', 'rnum', 'sido', 'yyyymm']
    tour_table = pd.DataFrame(jsonTP, columns=['yyyymm', 'resNm', 'ForNum'])
    tour_table = tour_table.set_index('yyyymm')     # set_index는 column을 index로 변환
    # print(tour_table)
    
    resNm = tour_table.resNm.unique()
    print('관광지명 : ', resNm[:5])     # '창덕궁' '운현궁' '경복궁' '창경궁' '종묘'
    
    # 중국인
    cdf = '../testdata/중국인방문객.json'
    jdata1 = json.loads(open(cdf, 'r', encoding='utf-8').read())
    china_table = pd.DataFrame(jdata1, columns=['yyyymm', 'visit_cnt'])
    china_table = china_table.rename(columns={'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    # print(china_table[:2])
    
    # 일본인
    jdf= '../testdata/일본인방문객.json'
    jdata2 = json.loads(open(jdf, 'r', encoding='utf-8').read())
    japan_table = pd.DataFrame(jdata2, columns=['yyyymm', 'visit_cnt'])
    japan_table = japan_table.rename(columns={'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    # print(japan_table[:2])
    
    # 미국인
    udf = '../testdata/미국인방문객.json'
    jdata3 = json.loads(open(udf, 'r', encoding='utf-8').read())
    usa_table = pd.DataFrame(jdata3, columns=['yyyymm', 'visit_cnt'])
    usa_table = usa_table.rename(columns={'visit_cnt':'usa'})
    usa_table = usa_table.set_index('yyyymm')
    # print(usa_table[:2])
    
    # 세 나라 관광객 수 합치기
    all_table = pd.merge(china_table,japan_table, left_index=True, right_index=True)
    all_table = pd.merge(all_table,usa_table, left_index=True, right_index=True)
    print(all_table[:3], all_table.shape)   # (72, 3)
    
    r_list = []
    for tp in resNm[:5]:
        r_list.append(setScatterChart(tour_table, all_table, tp))
        
    r_df = pd.DataFrame(r_list, columns=('고궁명', '중국', '일본', '미국'))
    r_df = r_df.set_index('고궁명')
    print(r_df)
    
    r_df.plot(kind='bar', rot=50)
    plt.show()
    
if __name__ == '__main__':
    start()
    



