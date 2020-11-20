# -*- coding: utf-8 -*-
# """
# Created on Thu Nov 19 12:45:46 2020

# @author: prtk9
# """

import numpy as np
import pandas as pd

oos = pd.read_csv('July_2019_OOS.csv')
oos = oos.drop_duplicates()
oos.columns

deletevars = ['Date','Machine_ID','Retailer_ID','Terminal_ID','TMID']
oos = oos.drop(deletevars, axis=1)
categoricalvars = ['termType', 'termType', 'Location_City','Zip']
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto', drop='first', handle_unknown='error', sparse=False, dtype=int)
ooscat = pd.DataFrame(ohe.fit_transform(oos[categoricalvars]), columns=ohe.get_feature_names())
ooscat = pd.concat([oos, ooscat], axis=1)
ooscat = ooscat.drop(categoricalvars, axis=1)
ooscat.head()

X = ooscat.drop(['Scratchers_Sales'], axis=1)
y = ooscat['Scratchers_Sales']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor   
ols = LinearRegression()
rfr = RandomForestRegressor(n_estimators=10, random_state=1)
gbr = GradientBoostingRegressor(random_state=0)
ergr = VotingRegressor(estimators=[('ols', ols), ('rfr', rfr), ('gbr', gbr)])
ergr.fit(X_train, y_train)
y_pred = ergr.predict(X_test)
print (ergr.score(X_test, y_test))