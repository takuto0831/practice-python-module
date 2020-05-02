# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 17:49:44 2017

@author: Rstudio
"""
import tensorflow as tf

_x = tf.constant(1)
_b = tf.constant(2)
res = tf.square(_x)
res = tf.add(res, _b)

# tool1
ses = tf.Session()
ses.run(res)

# tool2
with tf.Session() as sess:
    result = sess.run(y)
    print(result)

x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx = tf.constant([1, 0, 2])

idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
# tf.gather: dataのindexに従って値を取り出す
# tf.reshape: -1を指定すると行ベクトルとなる
y = tf.gather(tf.reshape(x, [-1]),  # flatten input
              idx_flattened)  # use flattened indices

ses.run(tf.reshape(x, [-1]))
ses.run(idx_flattened)
ses.run(tf.reshape(x, [9,1]))
