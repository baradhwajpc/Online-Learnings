# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:32:40 2018

@author: baradhwaj
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


os.chdir('G:\Python\Online-Learnings\Coursera\Linear Regression\Multi Variable Regression')
housePrice = pd.read_csv('HousePriceValues.txt',header=None,names=['Size', 'Bedrooms', 'Price'])
housePrice.head()

# Plot scatter 
plt.scatter('Size','Price',data=housePrice)
plt.scatter('Bedrooms','Price',data=housePrice)

axes = pd.tools.plotting.scatter_matrix(housePrice, alpha=0.5)

housePrice['Size'].corr(housePrice['Price']) # 0.85 - Higly correlated
housePrice['Bedrooms'].corr(housePrice['Price']) #0.44 - Non Correlated
housePriceModel = smf.ols(formula = 'Price ~ Size + Bedrooms',
                      data = housePrice).fit()
housePriceModel.summary()
# Adj R ^2 : 0.721 
# Size p value is insignificant - Not Negleting
# Bedrooms p value is significant   - Negletting
# 89600 + 139.2107 * Size -8738.0191 Bedrooms

