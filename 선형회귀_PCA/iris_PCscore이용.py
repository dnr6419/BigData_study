from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris() # 기본적으로 주어지는 datasets 중 하나
# dir(iris) # 어떠한 객체들이 포함되는지 확인 가능.
X = iris.data[:,[0,2]] # 독립변수 두개만을 설정 [row,column]
Y = iris.target
# print(X.shape)
# 독립변수 두개만 추출한 것에 대한 이름을 저장.
feature_names = [iris.feature_names[0],iris.feature_names[2]]
df_X = pd.DataFrame(X) # dataframe으로 저장.
df_Y = pd.DataFrame(Y)
# 결측치 존재여부 확인.
print(df_X.isnull().sum())
print(df_Y.isnull().sum()) 

pca = PCA(n_components=2) # dimension reduction 
pca.fit(X) 


# pca.explained_variance_ #eigen value 값을 나열하고 있음.
# pca.components_ # eigen vector값을 가지고 있음.
PCscore = pca.transform(X)
# PCscore 이용.

plt.scatter(PCscore[:,0],PCscore[:,1])
plt.show()
# 정사영 후의 데이터 븐포
# PC1 (왼쪽) 급격히 증가하는 것을 볼 수 있다. 많은 데이터를 설명할 수 있다.
# PC2 (오른쪽) 적게 설명하는 것을 알 수 있다.
