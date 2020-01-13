import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# 데이터 불러오기
boston = pd.read_csv("./Boston_house.csv")
# boston_data = boston.drop(['Target'],axis=1)
# 종속변수 설정
target = boston[['Target']]
# 모든 독립변수 설정
x_full = boston[['CRIM','RM','LSTAT','B','TAX','AGE','ZN','NOX','INDUS']]
y = target
x = x_full
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state = 1)
# train size와 test size로 나눠서 데이터 분할
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
