import pandas as pd
import numpy as np
import time 

# Matrix Factorizationを行う関数
def matrix_factorization(R, K, step=5000, alpha=0.0002, beta = 0.02):
    # R: n*m, P:n*k, Q:m*k
    # PとQの初期値を一様乱数により発生させる
    P = np.random.rand(len(R), K)
    Q = np.random.rand(K, len(R.T))
    #stes数繰り返す
    for step in range(step):
        for i in range(len(R)):
            for j in range(len(R.T)):
                # ユーザー評価が0の値を使わないため
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    # アルゴリズムに従って従って更新する
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
    return np.dot(P,Q)

# 誤差についての関数
def check_error(predict,test):
    # 誤差の行列
    error = predict - test
    # 値が0でない部分を取り除く
    i, j = test.nonzero()
    # 平均二乗誤差を返す
    return np.sqrt((error[i,j] ** 2).mean())

# 訓練データ、次元kについて誤差を求める関数
def matrix_check(num):
    # 訓練データの番号を記録する
    err_tmp = np.array([num+1])
    for i in range(10,31,10):
        print(str(num+1) + "番目" + "K = " + str(i) + "\n")
        # 各Kに対して予測値を算出する
        pred = matrix_factorization(train_mat[num], K = i)
        # 誤差率
        err = check_error(pred, test_mat[num])
        # 誤差平均を格納する
        err_tmp = np.append(err_tmp, err)
    # 訓練データの番号と各Kに対する誤差率を返す
    return err_tmp

# 実行時間測定
st = time.clock()
pred = matrix_factorization(train_mat[0], K = 30)
err = check_error(pred, test_mat[0])
en = time.clock() - st
print("processing time is %f second" % en)