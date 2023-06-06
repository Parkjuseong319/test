# numpy lib(module) - n-demensional array(다차원 배열), 수치해석, 선형대수, 벡터연산, 수학관련 등등..
# List 요소값을 바탕으로 평균, 합, 분산, 표준편차 구하기 1

datas = [1, 3, -2, 4]   # 변량 : 숫자로 나타낼 수 있는 자료

def show_data(datas):
    for d in datas:
        print(d, end = " ")
        
show_data(datas)

def show_sum(datas):
    tot = 0
    for d in datas:
        tot += d
    return tot

def show_avg(datas):    # 평균 : 산술, 기하, 조화, 제곱, ...(조화평균 나중에 알아보기)
    tot = show_sum(datas)
    avg = tot / len(datas)
    return avg

def show_variance(datas):
    avg = show_avg(datas)
    vari = 0
    for su in datas:
        vari += (su - avg) ** 2     # 편차 제곱의 합
    return vari / len(datas)    # 모집단으로 계산
    # return vari / (len(datas) - 1)    # 표본집단으로 계산

def show_sd(datas):
    return show_variance(datas) ** 0.5

print("\n 합은 ", show_sum(datas))
print("평균은 ", show_avg(datas))
print("분산은 ", show_variance(datas))
print("표준편차는 ", show_sd(datas))

print("numpy 모듈로 연산----전문가들에 의해 복잡하고 다양한 수식이 함수로 제공----")
import numpy as np

print("합은 ", np.sum(datas))
print("평균은 ", np.mean(datas))       # 결측값 무시하고 계산함. 일반적인 평균치
print("분산은 ", np.var(datas))
print("표준편차는 ", np.std(datas))
# print()
print("평균 두번째 ", np.average(datas))     # mean과 average함수는 좀 다르다.




