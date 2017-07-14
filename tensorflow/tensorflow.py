# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:57:03 2017

@author: SHIO-160412-4
"""
import tensorflow as tf
import numpy as np
import edward as ed

from edward.models import Normal
# X_train:特徴量のデータセット
#Y_train: 説明変数のデータセット

N = X_train.shape[0]  # データ数
D = X_train.shape[1] # 特徴数
M = 50000 #バッチサイズ
n_epoch=30 #ポック数

def four_layer_nn(x,W_1,W_2,W_3,b_1,b_2):
    h = tf.tanh(tf.matmul(x, W_1) + b_1)
    h = tf.tanh(tf.matmul(h, W_2) + b_2)
    h = tf.matmul(h, W_3) 
    return tf.reshape(h, [-1])

from edward.models import Normal, Bernoulli, Uniform

#事前分布 on weights and biases
W_1 = Normal(mu=tf.zeros([D, 20]), sigma=tf.ones([D, 20])*100)
W_2 = Normal(mu=tf.zeros([20, 15]), sigma=tf.ones([20, 15])*100)
W_3 = Normal(mu=tf.zeros([15, 1]), sigma=tf.ones([15, 1])*100)
b_1 = Normal(mu=tf.zeros(20), sigma=tf.ones(20)*100)
b_2 = Normal(mu=tf.zeros(15), sigma=tf.ones(15)*100)

#各ウェイトを近似する分布
qW_1 = Normal(mu=tf.Variable(tf.random_normal([D, 20])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, 20]))))
qW_2 = Normal(mu=tf.Variable(tf.random_normal([20, 15])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([20, 15]))))
qW_3 = Normal(mu=tf.Variable(tf.random_normal([15, 1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([15, 1]))))
qb_1 = Normal(mu=tf.Variable(tf.random_normal([20])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([20]))))
qb_2 = Normal(mu=tf.Variable(tf.random_normal([15])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([15]))))