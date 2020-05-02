import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""""
user - item の評価matrixを作成する関数
""""

# データ読み込み関数
def Read(num,file):
    # num:番号,file:base or test
    return pd.read_table("data/u{0}.".format(num) + file, header=None, usecols = [0,1,2]) 

# 行(user),列(item)に評価を格納する関数
def ReadMatrix(num):
    # num:番号,file:train or tesst
    for i in range(len(train[num])):
        train_mat[num, int(train[num,i,0] - 1), int(train[num, i, 1] - 1)] \
        = int(train[num, i, 2])
    for i in range(len(test[num])):
        test_mat[num, int(test[num, i, 0] - 1), int(test[num, i, 1] - 1)] \
        = int(test[num, i, 2])
         
# 3次元配列を用意する
train = np.empty((5,80000,3), dtype=int)
test = np.empty((5,20000,3), dtype=int)

# users:943人, items:1682個
train_mat = np.zeros(((5,943,1682)), dtype=int)
test_mat = np.zeros(((5,943,1682)), dtype=int)

# 訓練データとテストデータをそれぞれ格納する
# user id, item id, rating
for i in range(1,6):
    train[i-1] = Read(i, "base")
    test[i-1] = Read(i, "test")    
    
# 行(user),列(item)に評価を格納する
for i in range(5):
    ReadMatrix(i)
    
# rで操作するための書き出し
np.savetxt('csv/train1.csv',train_mat[0],delimiter=',', fmt='%.d')
np.savetxt('csv/test1.csv',test_mat[0],delimiter=',', fmt='%.d')

"""
tensorflowによるmatrix factorizationを行うために
index を取り出す関数
"""
def ReadIndex(train,test,num):
    X_train = train[num,:,[0,1]].T - 1
    X_test = test[num,:,[0,1]].T - 1
    y_train = train[num,:,2]
    y_test = test[num,:,2]
    return X_train, X_test, y_train, y_test
# 行(user),列(item)に評価を格納する
X_train, X_test, y_train, y_test = ReadIndex(train, test, 4)

"""   
bayesian network で扱えるようにユーザー情報、映画情報を属性、ジャンルで表す。    
"""
# データ形成形成関数
def MovieInfo():
    #訓練データとテストデータを生成し、csv形式で保存する 
    for j in range(5):
        # train data
        user = train[j,:,0] - 1
        item = train[j,:,1] - 1
        movie_data = np.c_[user_data[user], item_data[item], train[j,:,2]]
        df = pd.DataFrame(movie_data)
        df.to_csv("csv/train" + str(j+1) + ".csv", index=False, header = all_info)
        # test data
        user = test[j,:,0] - 1
        item = test[j,:,1] - 1
        movie_data = np.c_[user_data[user], item_data[item], test[j,:,2]]
        df = pd.DataFrame(movie_data)
        df.to_csv("csv/test" + str(j+1) + ".csv", index=False, header = all_info)

def MovieInfo2():
    #訓練データとテストデータを生成し、csv形式で保存する 
    #ageをpd.cutによって分割する
    for j in range(5):
        #train data
        user = train[j,:,0] - 1
        item = train[j,:,1] - 1
        movie_data = np.c_[user_data[user], item_data[item], train[j,:,2]]
        df = pd.DataFrame(movie_data)
        label = [str(x) for x in range(0,100,10)]
        df.iloc[:,1] = pd.cut(df.iloc[:,1], bins = range(0,101,10),labels = label) # 年齢列
        df.to_csv("csv/cut_data_10/train" + str(j+1) + ".csv", index=False, header = all_info)
        # test data
        user = test[j,:,0] - 1
        item = test[j,:,1] - 1
        movie_data = np.c_[user_data[user], item_data[item], test[j,:,2]]
        df = pd.DataFrame(movie_data)
        label = [str(x) for x in range(0,100,10)]
        df.iloc[:,1] = pd.cut(df.iloc[:,1], bins = range(0,101,10),labels = label) # 年齢列
        df.to_csv("csv/cut_data_10/test" + str(j+1) + ".csv", index=False, header = all_info)

# item data 読み込み 
item_data = pd.read_csv("data/u.item", header=None,delimiter="|",encoding = "ISO-8859-1") 
item_data = np.array(item_data)
item_data = np.delete(item_data, [1, 2, 3, 4], 1)
item_info = ["item-No", "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
"Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
# user data 読み込み
user_data = pd.read_csv("data/u.user", header=None,delimiter="|",usecols=[0,1,2,3]) 
user_data = np.array(user_data)
user_info = ["user-No", "age", "gender", "occupation"]

# 列名を作成
all_info = user_info + item_info + ["rating"]

"""""""""""""""""""""""""""
MovieInfo:4.5second
MovieInfo2:4.1second
"""""""""""""""""""""""""""
import time
st = time.time()
MovieInfo2()
en = time.time() - st
