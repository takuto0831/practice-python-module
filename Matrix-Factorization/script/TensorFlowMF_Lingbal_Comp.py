# -*- coding: UTF-8 -*-
# import set
import pandas as pd
import numpy as np
import tensorflow as tf # version = 1.7.0
import edward as ed
from edward.models import Normal

# set Python random seed
random.seed(31)
# set NumPy random seed
np.random.seed(31)
# set TensorFlow random seed
tf.set_random_seed(31)
# set Edward random seed
ed.set_seed(31)
# 表示させる
ses = tf.Session()

# number set 
N = user_log_num
M = event_num
K = 30  # number of latent dimensions

tmp = log.groupby(["user_id","event_id"])[["action_type"]].max() # 最大値抽出
tmp = tmp.reset_index() # multi index解除
x_train = np.array(tmp[["user_id","event_id"]]-1, dtype=int) # Tensorflow index
y_train = np.array(tmp[["action_type"]], dtype=int).T[0,:] # Tensorflow value

# Prepare a placeholder: indexを入れていく場所
X_ph = tf.placeholder(tf.int32, [None, 2])
#x_train = tf.constant(x_train,dtype=tf.float32)
#y_train = tf.constant(y_train,dtype=tf.float32)

# Building the model: 標準正規分布(loc:平均, scale:分散)
U = Normal(loc = tf.zeros([N, K]), scale = tf.ones([N, K]))
V = Normal(loc = tf.zeros([M, K]), scale = tf.ones([M, K]))

# 平均: -, 分散: 1
y = Normal(loc = tf.reduce_sum(tf.gather(U, X_ph[:, 0]) * tf.gather(V, X_ph[:, 1]), axis=1), 
           scale = tf.ones_like(tf.reduce_sum(tf.gather(U, X_ph[:, 0]) * tf.gather(V, X_ph[:, 1]), axis=1)))

# Building the variational model: tf.softplus:分散が負の値にならない, tf.Variable:変数
qU = Normal(loc = tf.Variable(tf.random_normal([N, K])), 
            scale = tf.nn.softplus(tf.Variable(tf.random_normal([N, K]))))
qV = Normal(loc = tf.Variable(tf.random_normal([M, K])), 
            scale = tf.nn.softplus(tf.Variable(tf.random_normal([M, K]))))

# Inference: qU, qVを更新していく
inference = ed.KLqp({U: qU, V: qV}, data = {X_ph: x_train, y: y_train})
inference.run(n_iter = 10000)

###### Tensorflow -> numpy ######
# qU.eval()
###### メモリ不足 ########
# 得られた行列積を算出
# train_matrix = tf.matmul(qU,qV,transpose_b=True) # 70000/158392
# 変数を実行させる前にはこの処理が必要
# init = tf.global_variables_initializer()
# ses.run(init)
# ses.run(train_matrix)
