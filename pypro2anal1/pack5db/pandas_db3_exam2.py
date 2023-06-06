import MySQLdb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pack5db.pandas_db2jikwon import jik_ypay
plt.rc('font', family='malgun gothic')
import sys

try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
        
except Exception as e:
    print("connect error ", e)
    sys.exit()  # 프로그램 강제종료
    
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    
except Exception as e:
    print("handler error ", e)
    cursor.close()
    conn.close()
