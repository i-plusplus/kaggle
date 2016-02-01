import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import threading
import datetime

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('train_combined.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test_combined.csv', encoding="ISO-8859-1")

df_train = df_train.drop(['search_term_nouns','product_title_nouns','product_description_nouns','search_term_stripped','product_title_stripped','product_description_stripped','search_term','product_title','product_description'],axis=1)

df_test = df_test.drop(['search_term_nouns','product_title_nouns','product_description_nouns','search_term_stripped','product_title_stripped','product_description_stripped','search_term','product_title','product_description'],axis=1)


id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
print ("estimation start")
clf.fit(X_train, y_train)
print ("estimation end")
y_pred = clf.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

