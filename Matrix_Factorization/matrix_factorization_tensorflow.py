# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 01:47:00 2017

@author: Takuto Kotsubo
"""
### Probabilistic matrix factorization ###

# tensorflowはpipで入れないと最新バージョンにならない 

import tensorflow as tf
import edward as ed
import numpy as np
from edward.models import Normal

# 中身を見るために必要！！
ses = tf.Session()

# number set 
N = 943 # 行数
M = 1682 # 列数
K = 30  # number of latent dimensions

# Prepare a placeholder
# indexを入れていく場所
X_ph = tf.placeholder(tf.int32, [None, 2])

# Building the model
# 標準正規分布(loc:平均, scale:分散)
# U:943行30列の乱数、V:1682行30列の乱数
U = Normal(loc = tf.zeros([N, K]), scale = tf.ones([N, K]))
V = Normal(loc = tf.zeros([M, K]), scale = tf.ones([M, K]))
# 平均: -, 分散: 1
y = Normal(loc = tf.reduce_sum(tf.gather(U, X_ph[:, 0]) * tf.gather(V, X_ph[:, 1]), axis=1), 
           scale = tf.ones_like(tf.reduce_sum(tf.gather(U, X_ph[:, 0]) * tf.gather(V, X_ph[:, 1]), axis=1)))

# Building the variational model
# tf.softplus:分散が負の値にならない, tf.Variable:変数, 
qU = Normal(loc = tf.Variable(tf.random_normal([N, K])), 
            scale = tf.nn.softplus(tf.Variable(tf.random_normal([N, K]))))
qV = Normal(loc = tf.Variable(tf.random_normal([M, K])), 
            scale = tf.nn.softplus(tf.Variable(tf.random_normal([M, K]))))

# 変数を実行させる前にはこの処理が必要
#init = tf.global_variables_initializer()
#ses.run(init)
#ses.run(qU)

# Inference
# qU, qVを更新していく
inference = ed.KLqp({U: qU, V: qV}, data = {X_ph: X_train, y: y_train})
inference.run(n_iter = 1000)

# Evaluation
y_prior = y
y_post = Normal(loc = tf.reduce_sum(tf.gather(qU, X_ph[:, 0]) * tf.gather(qV, X_ph[:, 1]), axis=1), 
                scale = tf.ones_like(tf.reduce_sum(tf.gather(qU, X_ph[:, 0]) * tf.gather(qV, X_ph[:, 1]), axis=1)))

print("Mean squared error (prior): ", ed.evaluate("mean_squared_error", data = {X_ph: X_test, y_prior: y_test}))
print("Mean squared error (posterior): ", ed.evaluate("mean_squared_error", data = {X_ph: X_test, y_post: y_test}))

"""
numpyで調べてみる
b = np.random.randn(20000) #標準正規分布
b = np.random.randint(1,6,20000) #整数
c = np.sum((y_test - b) * (y_test - b))/20000
"""

# edward.evaluate関数調査
# これがないと予測した回答見れないよ。。。。
from edward.models import Bernoulli, Binomial, Categorical, \
                           Multinomial, OneHotCategorical
from edward.util import check_data, get_session  
import six

ses = get_session()
n_samples = 500

#事前分布か更新した事後分布を使う
data = {X_ph: X_test, y_prior: y_test}
data = {X_ph: X_test, y_post: y_test}

#ブラックボックス
keys = [key for key in six.iterkeys(data) if not
        isinstance(key, tf.Tensor) or "Placeholder" not in key.op.type]
output_key = keys[0]
feed_dict = {key: value for key, value in six.iteritems(data)
             if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type}
y_true = data[output_key]

# 予測値算出
y_pred = [ses.run(output_key, feed_dict) for _ in range(n_samples)]
# tf.cast:型変換
y_pred = tf.cast(tf.add_n(y_pred), y_pred[0].dtype)/tf.cast(n_samples, y_pred[0].dtype)
pred_ans = np.array(ses.run(y_pred))

# MSEを求める
ans = tf.reduce_mean(tf.square(y_pred - y_true))
ses.run(ans)
# mse = 0.89294451