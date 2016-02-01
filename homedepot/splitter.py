import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import threading
import datetime

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
#df_attr = pd.read_csv('attributes.csv')
df_pro_desc = pd.read_csv('product_descriptions.csv')

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

totalsize = df_all.shape[0]
t = 32
for i in range(0,t):
    values = df_all.values[int(totalsize*i/(t*1.0)):int(totalsize*(i+1)/(t*1.0))]
    df = pd.DataFrame(values, columns = df_all.columns.values)
    df.to_csv(str(i) + ".csv", index = False)

