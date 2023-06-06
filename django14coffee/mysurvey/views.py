from django.shortcuts import render, redirect
from mysurvey.models import Survey
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')

def surveyView(request):
    return render(request, 'survey.html')

def surveyProcess(request):
    # 1 : 입력자료 DB에 저장
    insertData(request)
    # 2 : 분석 결과 보기 창으로 이동
    return redirect('/surveyshow')

def surveyAnalysis(request):
    # 여를 대상으로 선호하는 커피전문점을 선택하는 설문지 작성 후 커피전문점에 대한 소비자 인식조사를 한다. 
    # 이 결과로 '성별에 따라 선호하는 커피브랜드에 차이'가 있는지를 검정하시오.
    # 교차분석(카이스퀘어 검정)
    # 귀무 : 성별에 따라 선호하는 커피브랜드에 차이가 없다.
    # 대립 : 성별에 따라 선호하는 커피브랜드에 차이가 있다.
    
    data = list(Survey.objects.all().values())
    # print(data, type(data))
    
    df = pd.DataFrame(data)
    df.dropna()
    # print(df)
    df.columns = ['no', '성별', '나이대', '선호 커피브랜드']
    
    # dummy 변수 작성
    df['gender_num'] = df['성별'].apply(lambda g:1 if g=='남' else 2)
    df['co_num'] = df['선호 커피브랜드'].apply(lambda c:1 if c=='스타벅스' else 2 if c=='커피빈' else 3 if c=='이디야' else 4)
    # print(df)
    # ctab = pd.crosstab(index=df['gender_num'], columns=df['co_num'])
    ctab = pd.crosstab(index=df['성별'], columns=df['선호 커피브랜드'])
    
    # 카이제곱 검정
    chi, p, _, _ = stats.chi2_contingency(observed=ctab)
    
    if p > 0.05:
        msg = 'p값이 {} >= α 이므로<br/> 성별에 따라 선호하는 커피브랜드에 차이가 없다. 귀무 채택(대립 기각)'.format(p)
    else:
        msg = 'p값이 {} < α 이므로<br/> 성별에 따라 선호하는 커피브랜드에 차이가 있다. 대립 채택(귀무 기각)'.format(p) 
    # 설문 건수
    count = len(df)
    
    # 커피브랜드별 선호 건수에 대한 차트(세로막대)를 출력하시오
    fig = plt.gcf()
    coffee_group = df.groupby(df['선호 커피브랜드'])['no'].count()
    coffee_group.plot.bar(subplots=True, color=['cyan', 'green'], width=0.5, rot=0)
    plt.xlabel('커피 브랜드')
    plt.ylabel('득표 수')
    plt.title('커피 브랜드별 선호 건 수')
    plt.grid()
    fig.savefig('django14coffee/mysurvey/static/images/coffee.png') 
    
    context = {'datas':ctab.to_html(border=1), 'msg':msg, 'count':count}      # to_html에는 border=1이 기본값이다.
    return render(request, 'list.html', context)

# -----------------
def insertData(request):
    if request.method == 'POST':
        # print(request.POST.get('gen'), request.POST.get('age'), request.POST.get('age'))
        Survey(
            gender = request.POST.get('gen'),
            age = request.POST.get('age'),
            co_survey = request.POST.get('coffee')
        ).save()
