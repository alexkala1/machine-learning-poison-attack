# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def read_data():
    names = ['State', 'Account Length', 'Area Code', 'Phone', 'Intl Plan', 'VMail Plan', 
             'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge','Eve Mins', 'Eve Calls',
             'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge', 'Intl Mins', 'Intl Calls', 
             'Intl Charge', 'CustServ Calls', 'Churn']
    
    """"" Insert the filename"""""
    churn_df = pd.read_csv('dataset1.csv', names=names)
    churn_df.head()

    # Convert categorical variables to numeric
    numeric = preprocessing.LabelEncoder()
    churn_df['Intl Plan'] = numeric.fit_transform(churn_df['Intl Plan'])
    churn_df['VMail Plan'] = numeric.fit_transform(churn_df['VMail Plan'])

    # Isolate target column after converting to numeric
    y = numeric.fit_transform(churn_df['Churn'])

    # We don't need these columns
    to_drop = ['State','Area Code','Phone','Churn']
    churn_feat_space = churn_df.drop(to_drop,axis=1)
    
    X = churn_feat_space.values.astype(np.float)

    # Normalize data by mean and standard deviation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X,y