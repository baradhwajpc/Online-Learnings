# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 08:29:50 2017

@author: baradhwaj
"""
#Python for Data Science Essential Training
import numpy as np
import pandas as pd
dfObj = pd.DataFrame(np.arange(36).reshape(6,6))
print(dfObj)
dfObjTwo = pd.DataFrame(np.arange(15).reshape(5,3))
print(dfObjTwo)
# Concat 2 df's
print(pd.concat([dfObj,dfObjTwo],axis=1))
print(pd.concat([dfObj,dfObjTwo]))
dfObj.drop([0,2])
dfObj.drop([0,2],axis=1)