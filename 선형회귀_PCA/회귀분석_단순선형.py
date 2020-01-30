import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# 데이터 불러오기
boston = pd.read_csv("./Boston_house.csv")
# boston_data = boston.drop(['Target'],axis=1)
# 독립변수 설정
rm = boston[['RM']]
# 종속변수 설정
target = boston[['Target']]
# model 설정 및 예측
rm1 = sm.add_constant(rm,has_constant='add')
model = sm.OLS(target,rm1)
fitted_model = model.fit()
pred = fitted_model.predict(rm1)
# 계수 출력
print(fitted_model.params)
# 적합시킨 직선 시각화
plt.yticks(fontname='Arial')
plt.scatter(rm,target,label='data')
plt.plot(rm,pred,label='result')
plt.legend()
plt.show()
# 잔차의 합 계산
print(np.sum(fitted_model.resid))
# rm모델 residual 시각화
'''
fitted_model.resid.plot()
plt.xlabel('residual_number')
plt.show()
'''
# 단순선형모델 요약
print(fitted_model.summary())


