import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import time

# 변수선택법에는 크게 3가지가 있다. Forward, Backward, Stepwise
ploan = pd.read_csv('./Personal Loan.csv')
# 데이터 가져오기
ploan_processed = ploan.dropna().drop(['ID','ZIP Code'],axis =1,inplace = False)
# 의미없는 변수 두개 제거
ploan_processed = sm.add_constant(ploan_processed,has_constant='add')
# 상수항 추가
feature_columns = ploan_processed.columns.difference(["Personal Loan"])
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan']
# split
train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y,train_size=0.7,test_size=0.3,random_state=42)

# 임계치
def cut_off(y,threshold):
    Y = y.copy() # copy함수를 사용하여 이전의 y값이 변화지 않게 함
    Y[Y>threshold]=1
    Y[Y<=threshold]=0
    return(Y.astype(int))
# 정확도
def acc(cfmat):
    return (cfmat[0,0]+cfmat[1,1])/(cfmat[0,0]+cfmat[0,1]+cfmat[1,0]+cfmat[1,1])
# 추가 import
from sklearn.linear_model import Ridge, Lasso, ElasticNet


# lasso 적합
ll = Lasso(alpha=0.01) # alpha값은 lambda값
ll.fit(train_x,train_y)
# 예측, confusionmatrix, acc계산
pred_y_lasso = ll.predict(test_x)
# 임계치만큼 자른다
pred_Y_lasso = cut_off(pred_y_lasso,0.5)
cfmat = confusion_matrix(test_y,pred_Y_lasso)
# 성능 측정
print("lasso성능 : ",acc(cfmat))

# ridge 적합
rr = Ridge(alpha=0.01)
rr.fit(train_x,train_y)
## ridge result
print("ridge 회귀계수 ",rr.coef_)
# 0에 가까울 뿐,0 을 갖진 않는다.

## ridge y예측, confusion matrix, acc계산 
pred_y_ridge = rr.predict(test_x)
# 임계치만큼 자른다
pred_Y_ridge = cut_off(pred_y_ridge,0.5)
cfmat = confusion_matrix(test_y,pred_Y_lasso)
# 성능 측정
print("ridge의 성능 : ",acc(cfmat))


# alpha값에 따른 lasso와 ridge 성능분석
alpha = np.logspace(-3,2,6) # lasso
data = []
acc_table=[]
for i, a in enumerate(alpha):
    lasso = Lasso(alpha=a).fit(train_x, train_y)
    data.append(pd.Series(np.hstack([lasso.intercept_, lasso.coef_])))
    pred_y = lasso.predict(test_x) # full model
    pred_y= cut_off(pred_y,0.5)
    cfmat = confusion_matrix(test_y, pred_y)
    acc_table.append((acc(cfmat)))
    
df_lasso = pd.DataFrame(data, index=alpha).T
acc_table_lasso = pd.DataFrame(acc_table, index=alpha).T

data = []
acc_table=[]
for i, a in enumerate(alpha): # ridge
    ridge = Ridge(alpha=a).fit(train_x, train_y)
    data.append(pd.Series(np.hstack([ridge.intercept_, ridge.coef_])))
    pred_y = ridge.predict(test_x) # full model
    pred_y= cut_off(pred_y,0.5)
    cfmat = confusion_matrix(test_y, pred_y)
    acc_table.append((acc(cfmat)))

    
df_ridge = pd.DataFrame(data, index=alpha).T
acc_table_ridge = pd.DataFrame(acc_table, index=alpha).T

# lambda값(alpha값 변화에 따른 회귀계수 축소 시각화

import matplotlib.pyplot as plt
ax1 = plt.subplot(121)
plt.semilogx(df_ridge.T)
plt.xticks(alpha)
plt.title('Ridge')

ax2 = plt.subplot(122)
plt.semilogx(df_lasso.T)
plt.xticks(alpha)
plt.title("Lasso")

plt.show()
