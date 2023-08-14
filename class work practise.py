import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv(r"C:\Users\kdata\Desktop\KODI WORK\1. NARESH\1. MORNING BATCH\N_Batch -- 10.00AM\3. Aug\14th\MLR\Investment.csv")

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, 4]

X= pd.get_dummies(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

slope = regressor.coef_
slope

cons = regressor.intercept_
cons

bias = regressor.score(X_train, y_train)
bias #95%

variance = regressor.score(X_test, y_test)
variance #95%


# **** we build the model so far

import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5]]

#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,5]]

#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3]]

#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

X_opt = X[:, [0,1,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

'''
# data sceince you will inform to the ceo 
pleas spend on research part and lets wait for the result 
- works thats good ( 3mont)
-- change the model with more (50) 500
'''













