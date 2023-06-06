from django.shortcuts import render, redirect
from my_hypo.models import SurveyData
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rc('font',family='malgun gothic')


# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')


def SurveyFunc(request):
    if request.method == "GET":
        return render(request, 'survey.html')        
    elif request.method == "POST":
        # print(request.POST.get('job'))
        # print(request.POST.get('gender'))
        # print(request.POST.get('game_time'))
        SurveyData(
            job = request.POST.get('job'),
            gender = request.POST.get('gender'),
            game_time = request.POST.get('game_time')
        ).save()
        
        return redirect('/result')
        # return HttpResponseRedirect('/result')        

def ResultFunc(request):
    df = pd.DataFrame.from_records(SurveyData.objects.all().values())
    # 검정 1 : 성별에 따라  게임 사용시간 평균에 차이가 있는가?  <== t-test
    pval1 = stats.ttest_ind(df.loc[df['gender']=='남자','game_time'], df.loc[df['gender']=='여자','game_time'],\
                            equal_var=True)[1]  # 0번째 statistic, 1번째 p-value
    print("pval1 : ", pval1)
    result1= ""
    if pval1 > 0.05:
        result1 = 'p-value:%.5f > 0.05 이므로 성별에 따른 게임 이용시간에 차이가 없다.'%pval1
    elif pval1 < 0.05:
        result1 = 'p-value:%.5f <= 0.05 이므로 성별에 따른 게임 이용시간에 차이가 있다.'%pval1
    #  - 성별 자료 차트(세로막대)
    plt.figure()
    plot_df1 = pd.DataFrame(columns=['이용시간 평균'], index=['남자', '여자'])
    plot_df1.loc['남자'] = df.loc[df['gender']=='남자', 'game_time'].mean()
    plot_df1.loc['여자'] = df.loc[df['gender']=='여자', 'game_time'].mean()
    plot_df1.plot(kind='bar')
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\images\\result1.png')
    # os.path.dirname(os.path.realpath(__file__)) = 현재 파일 경로.
    plt.close()
    
    # 검정 2 : 직업에 따라  게임 사용시간 평균에 차이가 있는가?  <== oneway ANOVA
    _, pval2 = stats.f_oneway(df.loc[df['job']=='화이트칼라', 'game_time'],\
                              df.loc[df['job']=='블루칼라', 'game_time'],\
                              df.loc[df['job']=='학생', 'game_time'],\
                              df.loc[df['job']=='기타', 'game_time'])
    result2 = ""
    if pval2 > 0.05:
        result2 = 'p-value:%.5f > 0.05 이므로 직업에 따른 게임 이용시간에 차이가 없다.'%pval2
    elif pval2 < 0.05:
        result2 = 'p-value:%.5f <= 0.05 이므로 직업에 따른 게임 이용시간에 차이가 있다.'%pval2
    #  - 직업별 자료 차트(파이)
    plt.figure()
    plot_df2 = pd.DataFrame(columns=['이용시간 평균'], index=['화이트칼라', '블루칼라', '학생', '기타'])
    plot_df2.loc['화이트칼라'] = df.loc[df['job']=='화이트칼라', 'game_time'].mean()
    plot_df2.loc['블루칼라'] = df.loc[df['job']=='블루칼라', 'game_time'].mean()
    plot_df2.loc['학생'] = df.loc[df['job']=='학생', 'game_time'].mean()
    plot_df2.loc['기타'] = df.loc[df['job']=='기타', 'game_time'].mean()
    plot_df2.plot(kind='pie', subplots=True)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\images\\result2.png')
    plt.close()
    
    return render(request, 'result.html', {'result1':result1, 'result2':result2})