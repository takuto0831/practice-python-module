# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 22:28:17 2017

@author: SHIO-160412-4
"""
import numpy as np
from sympy import *
import seaborn as sns
import matplotlib.pyplot as plt

H = np.array([[1,-1,-1],
              [-1,2,3],
              [-1,3,5]])

# 初期値
start_point = np.array([[0],[0],[0]])

# Armijo条件
def armijo_method(x,d,H):
    g,t = 0.1,0.5 # 初期条件
    b = 1
    while f(x+b*d,H) >= f(x,H) + g*b*(np.dot(df(x,H).T,d)):
        b = b*t
    return b

# 最急降下法
def gradient_descent_method(point,H):
    eps  = 1e-5 # 終了条件
    tmp = point # 点の移動用
    history = point # 数値を記録
    iteration = 1000 # 最大繰り返し回数

    for i in range(iteration):
        d = -df(tmp,H) # 探索方向
        alpha = armijo_method(tmp,d,H) # ステップ幅
        tmp_new = tmp + alpha*d
        if np.abs(np.linalg.norm(tmp-tmp_new)) < eps:
            break
        tmp = tmp_new
        history = np.concatenate([history,tmp],axis = 1)
    return history.T

# 関数
def f(x,H):
    return x.sum() - 0.5*np.dot(x.T,np.dot(H,x))

# 導関数
def df(x,H):
    return np.array([[1],[1],[1]]) - np.dot(H,x)

data_gradient = gradient_descent_method(start_point,H)
