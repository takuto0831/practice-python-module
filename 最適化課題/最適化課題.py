# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:38:54 2017

@author: SHIO-160412-4
"""
import numpy as np
from sympy import *
import seaborn as sns
import matplotlib.pyplot as plt
# 関数
def f(x):
    return 4*x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 - 2*x[0] - 4*x[1]

# 導関数
def df(x):
    return np.array([8*x[0] + 2*x[1] -2,
                     2*x[0] + 4*x[1] -4])
# ヘッセ行列
def hesse():
    return np.array([[8.0,2.0]
                     [2.0,4.0]])
    
# Armijo条件
def armijo_method(x,d):
    g,t = 0.1,0.5 # 初期条件
    b = 1
    while f(x+b*d) >= f(x) + g*b*(np.dot(df(x).T,d)):
        b = b*t
    return b

# BFGS公式
def BFGS(beta,tmp_new,tmp):
    s = tmp_new - tmp
    y = df(tmp_new) - df(tmp)
    B_s,s_B_s,s_y = np.dot(beta,s),np.dot(np.dot(s.T,beta),s),np.dot(s.T,y)
    tmp_beta = beta - np.dot(B_s,B_s.T)/s_B_s + np.dot(y,y.T)/s_y
    return tmp_beta
 
# 初期値
start_point = np.array([[-1],[3]])
# 正定値対称な初期行列
start_beta = np.array([[1,0],[0,1]])

# 最急降下法
def gradient_descent_method(point):
    eps  = 1e-5 # 終了条件
    tmp = point # 点の移動用
    history = point # 数値を記録
    iteration = 1000 # 最大繰り返し回数

    for i in range(iteration):
        d = -df(tmp) # 探索方向
        alpha = armijo_method(tmp,d) # ステップ幅
        tmp_new = tmp + alpha*d
        if np.abs(np.linalg.norm(tmp-tmp_new)) < eps:
            break
        tmp = tmp_new
        history = np.concatenate([history,tmp],axis = 1)
    return history.T

# ニュートン法
def newton_method(point):
    eps  = 1e-5 # 終了条件
    tmp = point # 点の移動用
    history = point # 数値を記録
    iteration = 1000 # 最大繰り返し回数
    for i in range(iteration):
        d = - np.dot(np.linalg.inv(hesse(tmp)),df(tmp)) # 探索方向
        tmp_new = tmp + d
        if np.abs(np.linalg.norm(tmp-tmp_new)) < eps:
            break
        tmp = tmp_new
        history = np.concatenate([history,tmp],axis = 1)        
    return history.T

# 準ニュートン法
def quasi_newton_method(point,beta):
    eps  = 1e-5 # 終了条件
    tmp = point # 点の移動用
    tmp_b = beta # 正定値行列の変更用

    history = point # 数値を記録
    iteration = 1000 # 最大繰り返し回数
    
    for i in range(iteration):
        d = - np.dot(np.linalg.inv(tmp_b),df(tmp)) # 探索方向
        alpha = armijo_method(tmp,d) # ステップ幅
        tmp_new = tmp + alpha*d
        if np.abs(np.linalg.norm(tmp-tmp_new)) < eps:
            break
        tmp_b = BFGS(tmp_b,tmp_new,tmp) # betaの更新
        tmp = tmp_new
        history = np.concatenate([history,tmp],axis = 1)        
    return history.T
from fractions import Fraction
# 準ニュートン法
def quasi_newton_method_real(point,beta):
    eps  = 1e-5 # 終了条件
    tmp = point # 点の移動用
    tmp_b = beta # 正定値行列の変更用
    A = np.array([[4,1],
                  [1,2]])     
    history = point # 数値を記録
    iteration = 1 # 最大繰り返し回数
    
    for i in range(iteration):
        d = - np.dot(np.linalg.inv(tmp_b),df(tmp)) # 探索方向
        alpha = - np.dot(d.T,df(tmp)) / np.dot(np.dot(d.T,A),d) # ステップ幅
        tmp_new = tmp + alpha*d
        if np.abs(np.linalg.norm(tmp-tmp_new)) < eps:
            break
        tmp_b = BFGS(tmp_b,tmp_new,tmp) # betaの更新
        tmp = tmp_new
        history = np.concatenate([history,tmp],axis = 1)        
    return history.T



data_gradient = gradient_descent_method(start_point)
data_newton = newton_method(start_point)
data_quasi_newton = quasi_newton_method(start_point,start_beta)

sns.plt.title('gradient descent')
sns.pointplot(data_gradient[:,0],data_gradient[:,1],color="g",markers=[''])
sns.plt.title('newton method')
sns.pointplot(data_newton[:,0],data_newton[:,1],color="g")
sns.plt.title('quasi newton method')
sns.pointplot(data_quasi_newton[:,0],data_quasi_newton[:,1],color="g")

for i in range(len(data_gradient)):
    plt.plot(i,f(data_gradient[i]),"o")
for i in range(len(data_newton)):
    plt.plot(i,f(data_newton[i]),"o")
for i in range(len(data_quasi_newton)):
    plt.plot(i,f(data_quasi_newton[i]),"o")
    
    