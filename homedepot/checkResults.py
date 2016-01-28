from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import math
import sys
for i in range(1,len(sys.argv)):
    a = sys.argv[i]
    df_output = pd.read_csv(a )
    df_output = df_output['relevance'].values
    df_test = pd.read_csv('test_y.csv')
    df_actual = df_test['relevance'].values
    print (a + ' ' + str(mean_squared_error(df_actual,df_output)**0.5))


