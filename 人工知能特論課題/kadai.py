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

# 関数
def f(x,H):
    x
    return 4*x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 - 2*x[0] - 4*x[1]

# 導関数
def df(x):