from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris() # 기본적으로 주어지는 datasets 중 하나

X = iris.data
Y = iris.target
pca = PCA(n_components=4)  # dimention을 4로
pca.fit(X)

print(pca.explained_variance_)
# 첫번째와 두번째 PC가 설명력이 높은 것을 알 수 있음.

PCs=pca.transform(X)[:,0:2]
# PCscore 가져옴.
# 많아야 두개 적어도 1개를 받을 것을 위에서 확인했기 때문

cif = LogisticRegression(solver="sag",multi_class="multinomial").fit(PCs,Y)
# 두개만 뽑아서 로지스틱 회귀분석의 beta값을 구함.
# PCs를 이용하는 이유는 정사영한 데이터 값이 훨씬 설명이 잘 되기 때문.

print(confusion_matrix(Y,cif.predict(PCs)))
# 정확한지 여부를 확인.

clf2 = LogisticRegression(solver='sag', max_iter=1000, random_state=0,
                             multi_class="multinomial").fit(X[:,0:2], Y)
print(confusion_matrix(Y, clf2.predict(X2[:,0:2])))
# PCs를 사용하는 것이 X2변수를 이용하는 것보다 훨씬 정확한 예측을 보이는 것을 알 수 있음.
