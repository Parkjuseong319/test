# 한글 데이터에 대한 형태소 분석 라이브러리 사용
# 형태소(단어로써 의미를 갖는 최소단위) 분석 : 품사 분류를 통해 언어적 속성을 파악.
# pip install jpype1, pip install konlpy

from konlpy.tag import Kkma, Okt, Komoran

kkma = Kkma()
# 문장으로 잘라줌. list type으로 반환
print(kkma.sentences('한글 데이터 형태소 분석을 위한 라이브러리 설치를 합니다. 헬로우 에브리원'))  
# 명사만 얻음. list type으로 반환. 
print(kkma.nouns('한글데이터형태소분석을위한라이브러리설치를합니다'))
# 품사 태깅.
print(kkma.pos('한글데이터형태소분석을위한라이브러리설치를합니다'))
# 모든 품사로 분리.
print(kkma.morphs('한글데이터형태소분석을위한라이브러리설치를합니다'))

print()
okt = Okt()
# 명사만 얻음. list type으로 반환. 
print(okt.nouns('한글데이터형태소분석을위한라이브러리설치를합니다'))
# 품사 태깅.
print(okt.pos('한글데이터형태소분석을위한라이브러리설치를합니다'))
print(okt.pos('한글데이터형태소분석을위한라이브러리설치를합니다', stem=True))   # 원형(어근)으로 출력
print(okt.morphs('한글데이터형태소분석을위한라이브러리설치를합니다'))
# 어절 추출
print(okt.phrases('한글데이터형태소분석을위한라이브러리설치를합니다'))

print()
komo = Komoran()
# 명사만 얻음. list type으로 반환. 
print(komo.nouns('한글데이터형태소분석을위한라이브러리설치를합니다'))
# 품사 태깅.
print(komo.pos('한글데이터형태소분석을위한라이브러리설치를합니다'))
# 어절 추출
print(komo.morphs('한글데이터형태소분석을위한라이브러리설치를합니다'))
