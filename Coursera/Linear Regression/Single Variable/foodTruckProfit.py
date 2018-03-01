# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:12:30 2018

@author: baradhwaj
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Problem Statement : Predict profits for a food truck. 
#DV : Profit , IDV : Population

os.chdir('G:\Python\Online-Learnings\Coursera\Linear Regression\Single Variable')
profitData = pd.read_csv('singleLinearRegression.txt',header=None, names=['Population', 'Profit'])
profitData.head()

plt.scatter('Population','Profit',data=profitData)
profitData["Population"].corr(profitData["Profit"])
## Strong Positive correlation - 0.8378

profitLinearModel =  smf.ols('Profit ~ Population',data=profitData).fit()
profitLinearModel.summary()
# P value is significant - 0.000
# profit =  -3.8958 * population +1.1930
# R - squared : 0.702  - fairly good model

# Estimating the accuracy of the model

# Create a new df withoutt the dv column  - Profit
training_as_test = profitData.iloc[:,0:1]
predicted_profit_data =  -3.8958 * training_as_test['Population'] +1.1930
predicted_profit_data = profitLinearModel.predict(training_as_test)


percent_error = (profitData["Profit"] - predicted_profit_data)/profitData["Profit"] 
abs_percent_error = abs(profitData["Profit"] - predicted_profit_data)/profitData["Profit"] 

                        
                        
model_eval = pd.DataFrame({'Population':profitData["Population"],
                            'Actual Profit':profitData["Profit"],
                           'Predicted Profit':predicted_profit_data,
                           'Percetage Error':percent_error,
                          'Abs Percetage Error':abs_percent_error})

np.mean(model_eval['Abs Percetage Error']*100)


