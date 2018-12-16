# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:40:51 2017

@author: SHIO-160412-4
"""
import pandas as pd
import numpy as np
import dask.array as da
import dask.dataframe as dd
from joblib import Parallel, delayed

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                [9, 10, 11, 12], [13, 14, 15, 16]])
arrr = np.array([arr, arr])
darr = da.from_array(arrr, chunks=(1,4,4))
darr
darr.sum().compute()

names = ["user id", "item id", "rating"]

u1_train = dd.read_table("data/u1.base", names = names, usecols = [0,1,2])
u1_train.rating.count().compute()

# データ読み込み
def read(num,file):
    # num:番号,file:base or test
    return pd.read_table("data/u{0}.".format(num) + file, names = names, usecols = [0,1,2])


read(1,"base")



from multiprocessing import Pool
import multiprocessing as multi

p = Pool(multi.cpu_count())
p.map(process, list(range(100)))
p.close()

test.py
# -*- coding: utf-8 -*-
from multiprocessing import Pool

def fuga(x): # 並列実行したい関数
    return x*x

def hoge():
    p = Pool(8) # 8スレッドで実行
    p.map(fuga, range(10)) # fugaに0,1,..のそれぞれを与えて並列演算

if __name__ == "__main__":
    hoge()
    
import threading

def worker():
    print("WWW")
    return 

threads = []
for i in range(5):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()
    
#     
from multiprocessing import Process
import time

def countDown(n):
    while n > 0:
        n -= 1

n = int(1e9)
n1, n2, n3, n4 = int(n/4), int(n/4), int(n/4), int(n/4)

jobs = [
        Process(target=countDown, args=(n1,)),
        Process(target=countDown, args=(n2,)),
        Process(target=countDown, args=(n3,)),
        Process(target=countDown, args=(n4,)),
        ]

start_time = time.time()
for j in jobs:
    j.start()

for j in jobs:
    j.join()

finish_time = time.time()
print(finish_time - start_time)
# 0.9second

start_time = time.time()
countDown(n1)
finish_time = time.time()
print(finish_time - start_time)
# 14second

def MatrixPredict(ite):
    # for i in range(5,31,5):
    i = 5
    # 各Kに対して予測値を算出する
    pred = matrix_factorization(train_mat[ite], K = i)
    # 誤差率
    err = check_error(pred, test_mat)
    # 誤差率を返す
    return err

jobs = [
        Process(target=MatrixPredict, args=(0,)),
        Process(target=MatrixPredict, args=(1,)),
        Process(target=MatrixPredict, args=(2,)),
        Process(target=MatrixPredict, args=(3,)),
        Process(target=MatrixPredict, args=(4,)),
        ]

start_time = time.time()
for j in jobs:
    j.start()

for j in jobs:
    j.join()

finish_time = time.time()
print(finish_time - start_time)

from multiprocessing import Pool, cpu_count, current_process
import time

def slowf(x):
    print(current_process().name,
        ': This started at %s.' % time.ctime().split()[3])
    
    time.sleep(1)
    return x*x

if __name__ == '__main__':
    print('cpu : %d' % 1)
    st = time.time()
    print('answer : %d' % sum(map(slowf, range(10))))
    print('time : %.3f s' % (time.time()-st
    

    print('\ncpu : %d' % cpu_count())
    st = time.time()
    p = Pool()
    print('answer : %d' % sum(p.map(slowf, range(10))))
    print('time : %.3f s' % (time.time()-st))
    
if __name__ == '__main__':
    def process(i):
        return [{'id': j, 'sum': sum(range(i*j))} for j in range(10)]
    
    result = Parallel(n_jobs=3, verbose=10)([delayed(process)(n) for n in range(100)])