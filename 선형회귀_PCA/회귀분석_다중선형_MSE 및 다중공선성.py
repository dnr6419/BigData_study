import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 데이터 불러오기
boston = pd.read_csv("./Boston_house.csv")
# boston_data = boston.drop(['Target'],axis=1)
# 종속변수 설정
target = boston[['Target']]
# 독립변수 설정
x_data = boston[['CRIM','RM','LSTAT']]

# 상수항 추가 및 모델 적합
x_data1 = sm.add_constant(x_data,has_constant='add')
multi_model = sm.OLS(target,x_data1)
fitted_model = multi_model.fit()

# 다중공선성진단
x_full = boston[['CRIM','RM','LSTAT','B','TAX','AGE','ZN','NOX','INDUS']]
x_full1 = sm.add_constant(x_full,has_constant='add')
full_model = sm.OLS(target,x_full1)
fitted_full_model = full_model.fit()

# 상관계수/산점도를 통한 다중공선성 확인
print(x_full.corr()) # full 독립변수에서 corr계수확인

# VIF를 통한 다중공선성
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x_full.values,i) for i in range(x_full.shape[1])]
vif["features"] = x_full.columns
print(vif)
# vif가 10보다 크면 상관성이 낮은 걸로 인식

# MSE를 통한 검증
from sklearn.metrics import mean_squared_error

print("CRIM,RM,LSTAT 변수만 이용했을 때, MSE 값")
print(mean_squared_error(y_true=target,y_pred=fitted_model.predict(x_data1)))
print("모든 독립변수를 이용했을 때, MSE 값")
print(mean_squared_error(y_true=target,y_pred=fitted_full_model.predict(x_full1)))
