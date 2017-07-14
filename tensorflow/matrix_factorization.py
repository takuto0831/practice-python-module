# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 01:47:00 2017

@author: Rstudio
"""
### Probabilistic matrix factorization ###

import tensorflow as tf
import edward as ed
import numpy as np

from edward.models import Normal

ed.set_seed(42)

def build_dataset(U, V, n_samples):
    N = U.shape[0]
    M = V.shape[0]
    K = U.shape[1]
    X = []
    sampled = set()
    
    for _ in range(n_samples):
        i = np.random.randint(N)
        j = np.random.randint(N)
        while (i, j) in sampled:
            i = np.random.randint(N)
            j = np.random.randint(N)
        sampled.add((i, j))
        X.append([i, j])
        
    X = np.array(X)
    y = np.random.normal(loc = np.sum(U[X[:, 0]] * V[X[:, 1]], axis=1), scale = 1.0)
    
    return X, y

def train_test_split(X, y, test_ratio):
    n_samples = X.shape[0]
    test_indices = np.random.binomial(1, test_ratio, size = n_samples)
    return X[test_indices==0], y[test_indices==0], X[test_indices==1], y[test_indices==1]
    
N = 30
M = 40
K = 5  # number of latent dimensions
n_samples = 300
U_true = np.random.normal(loc = 0.0, scale = 1.0, size = (N, K))
V_true = np.random.normal(loc = 0.0, scale = 1.0, size = (M, K))
X_data, y_data = build_dataset(U_true,  V_true, n_samples)

test_ratio = 0.1
X_train, y_train, X_test, y_test = train_test_split(X_data, y_data, test_ratio)

# Prepare a placeholder

X_ph = tf.placeholder(tf.int32, [None, 2])

# Building the model

U = Normal(loc = tf.zeros([N, K]), scale = tf.ones([N, K]))
V = Normal(loc = tf.zeros([M, K]), scale = tf.ones([M, K]))
y = Normal(loc = tf.reduce_sum(tf.gather(U, X_ph[:, 0]) * tf.gather(V, X_ph[:, 1]), axis=1), scale = tf.ones_like(tf.reduce_sum(tf.gather(U, X_ph[:, 0]) * tf.gather(V, X_ph[:, 1]), axis=1)))

# Building the variational model

qU = Normal(loc = tf.Variable(tf.random_normal([N, K])), scale = tf.nn.softplus(tf.Variable(tf.random_normal([N, K]))))
qV = Normal(loc = tf.Variable(tf.random_normal([M, K])), scale = tf.nn.softplus(tf.Variable(tf.random_normal([M, K]))))

# Inference

inference = ed.KLqp({U: qU, V: qV}, data = {X_ph: X_train, y: y_train})
inference.run(n_iter = 1000)

# Evaluation

y_prior = y
y_post = Normal(loc = tf.reduce_sum(tf.gather(qU, X_ph[:, 0]) * tf.gather(qV, X_ph[:, 1]), axis=1), scale = tf.ones_like(tf.reduce_sum(tf.gather(qU, X_ph[:, 0]) * tf.gather(qV, X_ph[:, 1]), axis=1)))

print("Mean squared error (prior): ", ed.evaluate("mean_squared_error", data = {X_ph: X_test, y_prior: y_test}))
print("Mean squared error (posterior): ", ed.evaluate("mean_squared_error", data = {X_ph: X_test, y_post: y_test}))
