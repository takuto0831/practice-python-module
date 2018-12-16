# 提出しない！！！！！！！！！！！！！！！！！！！！！！！！！！

from sklearn.decomposition import NMF
import numpy as np

model = NMF(n_components=2, init='random', random_state=0) # n_componentsで特徴の次元を指定
P = model.fit_transform(train_mat[0]) # 学習
Q = model.components_

model.reconstructions_


