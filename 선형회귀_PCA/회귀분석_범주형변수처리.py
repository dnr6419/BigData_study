import os
import pandas as pd 
import numpy as np
import statsmodels.api as sm
# data 가져오
corolla = pd.read_csv('./ToyotaCorolla.csv')
# 데이터 수와 변수의 수 확인하기
nCar = corolla.shape[0]
nVar = corolla.shape[1]
# 명목형 변수의 전처리
corolla.Fuel_Type.unique()
#세개의 dummy variables
dummy_p = np.repeat(0,nCar)
dummy_d = np.repeat(0,nCar)
dummy_c = np.repeat(0,nCar)
# 인덱스 슬라이싱 후 1 대입.
p_idx = np.array(corolla.Fuel_Type == 'Petrol')
d_idx = np.array(corolla.Fuel_Type == 'Diesel')
c_idx = np.array(corolla.Fuel_Type == 'CNG')
dummy_p[p_idx] = 1
dummy_d[d_idx] = 1
dummy_c[c_idx] = 1
# 새로운 데이터프레임 Fuel을 만든다
# corolla data의 변수를 각각 지운다.
# corolla와 Fuel 데이터프레임을 concat으로 합친다.
Fuel = pd.DataFrame({'Petrol': dummy_p, 'Diesel': dummy_d, 'CNG': dummy_c})
corolla_ = corolla.drop(['Id','Model','Fuel_Type'],axis=1,inplace=False)
mir_data = pd.concat((corolla_, Fuel), 1)
