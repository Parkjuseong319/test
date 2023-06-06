# tensor와 상수, 변수

# const:1은 c1의 주소를 참조한다. c언어에서 포인터처럼 참조한다고 생각하면 된다. tensorflow는 c기반 라이브러리
# tensorflow는 그래프 영역내에서 작업된다.

# tensor 연산은 별도의 Graph 영역 내에서 작업한다.
import tensorflow as tf
from astropy.io.tests.mixin_columns import su
"""
g1 = tf.Graph()     # 별도의 그래프 명시적 선언.

with g1.as_default():       # 객체 하나를 반환한다.
    c1 = tf.constant(1, name='c_one')   # c1은 c_one 이란 이름을 가지고 정수 1을 가진 상수
    print(c1)
    print(type(c1))
    print(c1.op)            # op는 그래프 생성함수.
    print(g1.as_graph_def())        
print(g1)       # graph object임을 확인. 그래프 영역에서 따로 진행. 그래프 영역은 c 라이브러리가 기반으로 되어있다.

"""
import numpy as np      # tensorflow numpy 기반으로 되어있음

# tf.constant() : tensor(배열)을 직접 기억. 일반적으로 상수라고 부름
# tf.Variable() : tensor가 저장된 주소를 참조 

node1 = tf.constant(3, tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)
imsi = tf.add(node1, node2)     # numpy 기반이라 add 함수 존재
print(imsi)
print()
node3 = tf.Variable(3, dtype=tf.float32)
node4 = tf.Variable(4.0)
print(node3)
print(node4)
# node4.assign(node3)     # node4 주소에 node3을 참조
node4.assign_add(node3)     # node4 += node3 의 의미.
print(node4)
print()
a = tf.constant(5)
b = tf.constant(10)
c = tf.multiply(a, b)       # 곱
print(c, c.numpy())        # c.numpy를 했을때 그래프 영역이 아니라 일반 영역으로 나오게 된다.

result = tf.cond(a < b, lambda:tf.add(a,b), lambda:tf.square(a))
print(result.numpy())

# function 관련. 
v = tf.Variable(1)

@tf.function()          # auto graph 기능에 의해 Graph 객체 환경에서 작업함
def find_next_Func():       # tensor를 일반 함수로 돌리는건 비효율적이다. 그래서 autograph decoration 추가.
    v.assign(v + 1)
    if tf.equal(v % 2, 0):
        v.assign(v + 10)

find_next_Func()
print(v.numpy())
print(type(find_next_Func))
    
print('------'*15)
def func1():
    imsi = tf.constant(0)       # imsi = 0 도 가능. 일반함수이기 때문
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        # imsi = imsi + su
        imsi += su
        
    return imsi

kbs = func1()
print(kbs, kbs.numpy(), np.array(kbs))

print()
# imsi = tf.constant(0)
@tf.function()
def func2():
    imsi = tf.constant(0)       # constant는 함수 밖, 안 상관없이 위치해도 된다.
    # global imsi
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        imsi += su
        
    return imsi

mbc = func2()
print(mbc.numpy())

print('tf.Variable() ----------------------')
imsi = tf.Variable(0)

@tf.function()
def func3():
    # imsi = tf.Variable(0)       # Variable()은 함수내에서는 적용 되지만 글로벌 영역에 있으면 적용되지 않는다. 
    # autograph 기능이 있다면 Variable은 함수 안이 아닌 바깥에 있어야한다.
    su = 1
    for _ in range(3):
        imsi.assign_add(su)     # Variable로 선언 시 누적은 assign_add 사용.
        # imsi += su
    
    return imsi

sbs = func3()
print(sbs.numpy())

print('구구단 출력 --------------')

@tf.function()
def gugu1(dan):
    su = 0
    for _ in range(9):
        su = tf.add(su, 1)
        # print(su.numpy())        # autograph에서는 numpy()로 형변환 X
        # print('{} * {} = {:2}'. format(dan, su, dan * su))  # 서식이 있는 출력 X
        print(su)
gugu1(3)

print()
# @tf.function()
def gugu2(dan):
    for i in range(1,10):
        result = tf.multiply(dan, i)        # 원소끼리의 곱     # tf.matmul() 행렬의 곱
        print('{} * {} = {:2}'. format(dan, i , result))
            
gugu2(3)
