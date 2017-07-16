# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv as csv
import random
import tensorflow as tf
from edward.models import Normal, Bernoulli, Uniform
import edward as ed
from tqdm import tqdm
import time

train_df = pd.read_table("C:/Users/Rstudio/Desktop/vasily_test/csv/train1.csv",delimiter=",") 
test_df = pd.read_table("C:/Users/Rstudio/Desktop/vasily_test/csv/test1.csv",delimiter=",") 
# reindex:探索, reset_index:indexを0から振りなおす
train_df = train_df.reindex(np.random.permutation(train_df.index)).reset_index(drop=True)

Y_train = np.array(train_df)[:,24]
Y_train = np.reshape(Y_train,[-1,1])

#テストデータのの目的変数もまず-1を0に変換してから抽出
test_df.loc[(test_df.is_Higgs == -1 ),"is_Higgs"] = 0
Y_test = test_df.values[0::,0]

#トレーニングデータの説明変数の正規化
train_df_means = train_df.mean().values
train_df_max = train_df.max().values
train_df_min = train_df.min().values
X_train = train_df.apply(lambda x: (x-x.mean())/(x.max()-x.min()), axis=0).fillna(0).values[0::,1::]

#テストデータの説明変数を上の正規化定数を使って正規化
X_test = test_df.values[0::,1::]
for i in range(X_test.shape[1]):
	X_test[0::,i] = (X_test[0::,i] - train_df_means[i+1])/(train_df_max[i+1]-train_df_min[i+1] )

def four_layer_nn(x,W_1,W_2,W_3,b_1,b_2):
    h = tf.tanh(tf.matmul(x, W_1) + b_1)
    h = tf.tanh(tf.matmul(h, W_2) + b_2)
    h = tf.matmul(h, W_3) 
    return tf.reshape(h, [-1])

def make_batch_list(x_data,y_data, batch_size=1):
  data_array = np.concatenate((y_data,x_data), axis=1)
  n = data_array.shape[0]
  n_batches = n // batch_size
  x_batch_list = []
  y_batch_list = []
  np.random.shuffle(data_array)
  for i in range(n_batches):
    x_batch_list.append(data_array[i*batch_size:(i+1)*batch_size,1:])
    y_batch_list.append(data_array[i*batch_size:(i+1)*batch_size,0])
  return x_batch_list, y_batch_list

N = X_train.shape[0]  # データ数
D = X_train.shape[1] # 特徴数
M = 50000 #バッチサイズ
n_epoch=30 #ポック数

#事前分布 on weights and biases
W_1 = Normal(mu=tf.zeros([D, 20]), sigma=tf.ones([D, 20])*100)
W_2 = Normal(mu=tf.zeros([20, 15]), sigma=tf.ones([20, 15])*100)
W_3 = Normal(mu=tf.zeros([15, 1]), sigma=tf.ones([15, 1])*100)
b_1 = Normal(mu=tf.zeros(20), sigma=tf.ones(20)*100)
b_2 = Normal(mu=tf.zeros(15), sigma=tf.ones(15)*100)

#変分ベイズの仮定
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

#Inferenceクラスの構築. X_train/Y_trainのBatchを入れるplace_holderを準備
x_ph  = tf.placeholder(tf.float32,[M,D])
y = Bernoulli(logits=four_layer_nn(x_ph, W_1, W_2, W_3, b_1, b_2))
y_ph = tf.placeholder(tf.float32,[M])
inference = ed.KLqp({W_1: qW_1, b_1: qb_1,W_2: qW_2, b_2: qb_2,W_3: qW_3}, data={y: y_ph})

#推論開始(subsamplingについては:http://edwardlib.org/api/inference-data-subsampling)
# n_iterはこのセッションにおけるiteration回数.
# n_printはinfo_dictに出力する回数.
inference.initialize(n_iter=n_epoch*int(float(N)/M), n_print=100)
tf.initialize_all_variables().run()
X_batch_list, Y_batch_list = make_batch_list(X_train,Y_train,M)
for _ in tqdm(xrange(n_epoch)):
  for i in xrange(len(X_batch_list)):
    info_dict=inference.update(feed_dict={x_ph: X_batch_list[i], y_ph: Y_batch_list[i]})
    inference.print_progress(info_dict)
inference.finalize() #session終了

#評価(http://edwardlib.org/tutorials/point-evaluation)
#X_testを挿入するためのTFのplaceholderを準備.
x_test_ph  = tf.placeholder(tf.float32,[X_test.shape[0],D])
#次にy_postに推定された事後分布をコピー
y = Bernoulli(logits=four_layer_nn(x_test_ph, W_1, W_2, W_3, b_1, b_2))
y_post = ed.copy(y, {W_1: qW_1.mean(), b_1: qb_1.mean(),W_2: qW_2.mean(), \
 b_2: qb_2.mean(),W_3: qW_3.mean()})

#BinaryAccuacyで評価
print 'Binary accuracy on test data:'
print ed.evaluate('binary_accuracy', data={x_test_ph: X_test, y_post: Y_test})
