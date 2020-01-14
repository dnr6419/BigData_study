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
# 대출여부를 전처리
train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y,train_size=0.7,test_size=0.3,random_state=42)
# split
model = sm.Logit(train_y,train_x)
results=model.fit(method='newton')
print(results.summary())
# 로지스틱 회귀분석

print(np.exp(results.params))
# 로지스틱 회귀계수값 출력

def cut_off(y,threshold):
    Y = y.copy() # copy함수를 사용하여 이전의 y값이 변화지 않게 함
    Y[Y>threshold]=1
    Y[Y<=threshold]=0
    return(Y.astype(int))

pred_Y = cut_off(pred_y,0.5)
# 기본적으로 0.5를 임계치로 계산한다.
print(pred_Y)

# confusion matrix
# 혼동 행렬
cfmat = confusion_matrix(test_y,pred_Y)
print(cfmat)
# 정확도 계산하
def acc(cfmat):
    return (cfmat[0,0]+cfmat[1,1])/(cfmat[0,0]+cfmat[0,1]+cfmat[1,0]+cfmat[1,1])
acc(cfmat)
