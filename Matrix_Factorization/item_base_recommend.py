import numpy as np
import read_data

# 相関係数を計算する関数
def similarity_pearson(item1, item2):
    # item1, item2ともに評価されている部分を使う
    num = np.logical_and(item1 != 0, item2 != 0)
    # 共通部分の個数
    count = np.sum(num)
    # 共通する部分がない場合0を帰す
    if count == 0: return 0
    
    # 列ベクトルの和、二乗和を求める
    sum1 = np.sum(item1[num])
    sum2 = np.sum(item2[num])
    sum_sq1 = np.sum(item1[num] ** 2)
    sum_sq2 = np.sum(item2[num] ** 2)
    sum_sq12 = np.sum(item1[num] * item2[num])
    
    # 相関係数を求める
    S_xy = sum_sq12 - (sum1 * sum2 / count)
    S_xx = sum_sq1 - (sum1 ** 2 / count)
    S_yy = sum_sq2 - (sum2 ** 2 / count)
    sqr = np.sqrt(S_xx * S_yy)
    if sqr == 0: return 0
    r = S_xy / sqr
    # 相関係数を返す
    return r
    
# アイテムの類似度を求める関数
def item_similarity(R):
    # num: item数
    num = len(R.T)
    # 類似性を格納する
    sims = np.zeros((num,num))

    for i in range(num):
        for j in range(i, num):
            if i == j:
                # 対角成分は 1
                sim = 1.0
            else:
                # R[:, i] は アイテム i に関する全ユーザの評価を並べた列ベクトル
                sim = similarity_pearson(R[:,i], R[:,j])
            # 得られた値を(i,j),(j,i)に格納する
            sims[i][j] = sim 
            sims[j][i] = sim 

    return sims 

# アイテムの類似度を用いて予測を行う関数
def predict(R, sims):
    # 元のデータを保持する
    R1 = R
    # 未評価は0, 評価済は1となる行列
    x = np.zeros((len(R), len(R.T)))
    x[R > 0] = 1
    
    # Rの成分が０の場所に予測値を入れる
    for i in range(len(R)):
        for j in range(len(R.T)):
            if R[i,j] == 0:
                normarize = np.sum(np.abs(x[i] * sims[j]))
                # 正規化する値が0でないか確認する
                if normarize > 0:
                    R1[i,j] = np.dot(sims[j],R[i])/normarize
    # 予測値を含めた行列を返す
    return R1

# 5つのデータを用いて平均誤差を求める
err_all = np.array([])
# 類似データの取得,予測,誤差を求める計算を繰り返す
for i in range(5):
    sims = item_similarity(train_mat[i])
    pred = predict(train_mat[i], sims)
    err = check_error(pred,test_mat[i])
    err_all = np.append(err_all, err)

