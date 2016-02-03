import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import threading
import datetime
from xgboost.sklearn import XGBRegressor
from sklearn import grid_search, linear_model
from sklearn.linear_model import LinearRegression
stemmer = SnowballStemmer('english')

df_train_x = pd.read_csv('train_x.csv', encoding="ISO-8859-1")
df_test_x = pd.read_csv('test_x.csv', encoding="ISO-8859-1")

df_train_x = df_train_x.drop(['search_term_nouns','product_title_nouns','product_description_nouns','search_term_stripped','product_title_stripped','product_description_stripped','search_term','product_title','product_description'],axis=1)

df_test_x = df_test_x.drop(['search_term_nouns','product_title_nouns','product_description_nouns','search_term_stripped','product_title_stripped','product_description_stripped','search_term','product_title','product_description'],axis=1)

df_train_x.to_csv("/tmp/train.csv")

id_test = pd.read_csv('test_id.csv')

y_train = pd.read_csv('train_y.csv')

xgb = XGBRegressor(objective='reg:linear', subsample=1.0, colsample_bytree=0.5, seed=0)                  
parameters = {'max_depth' : [6], 'learning_rate':[.3], 'n_estimators':[40]}
clf = grid_search.GridSearchCV(xgb, parameters)
clf.fit(df_train_x.values, y_train['relevance'].values)

y_pred = clf.predict(df_test_x)

id_test['relevance'] = [max(x,1) for x in y_pred]
id_test['relevance'] = [min(x,3) for x in id_test['relevance'].values]


id_test.to_csv('submission.csv',index=False)

