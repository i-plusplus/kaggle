import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import threading
import datetime
import sys
stemmer = SnowballStemmer('english')
it = 0
def str_stemmer(s):
        global it
        it = it +1
        return " ".join([stemmer.stem(word) for word in s.lower().split()])

stop = stopwords.words('english')
def str_remove_stopwords(s):
        global it
        global stop
        it = it +1
        return " ".join([x for x in str(s).lower().split() if x not in stop])

def str_common_word(str1):
        global it
        it = it +1
        str2 = str(str1[1])
        str1 = str(str1[0])
        return sum(int(str2.find(word)>=0) for word in str1.split())

def str_nouns(s):
        global it
        it = it +1
        if it%100 == 0:
             print([str(it),sys.argv[1]])
             print(datetime.datetime.now())
        tagged_sent = pos_tag(s.split())
        return " ".join([word for word,pos in tagged_sent if pos == 'NNP' or pos == 'NN'])

df_all = pd.read_csv('test/' + sys.argv[1], encoding="ISO-8859-1")
print("nouns start")
it = 0
#df_all['search_term_nouns'] = df_all['search_term'].map(lambda x:str_nouns(x))
it = 0
#df_all.to_csv(sys.argv[1],index=False)
print("noun title start")
#df_all['product_title_nouns'] = df_all['product_title'].map(lambda x:str_nouns(x))
print("noun description start")
it = 0
#df_all.to_csv(sys.argv[1],index=False)
#df_all['product_description_nouns'] = df_all['product_description'].map(lambda x:str_nouns(x))
it = 0
print ("nouns fetched")
#df_all.to_csv(sys.argv[1],index=False)
df_all['search_term_stripped'] = df_all['search_term'].map(lambda x:str_remove_stopwords(x))
print("stripped title start")
it = 0
df_all.to_csv(sys.argv[1],index=False)
df_all['product_title_stripped'] = df_all['product_title'].map(lambda x:str_remove_stopwords(x))
print("stripped desc start")
it = 0
df_all.to_csv(sys.argv[1],index=False)
df_all['product_description_stripped'] = df_all['product_description'].map(lambda x:str_remove_stopwords(x))
print ("stripped fetched")
df_all.to_csv(sys.argv[1],index=False)
it = 0
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
df_all.to_csv(sys.argv[1],index=False)
print ("all data fetched")

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['word_in_title'] = df_all[['search_term','product_title']].apply(str_common_word, axis = 1)
df_all['word_in_description'] = df_all[['search_term','product_description']].apply(str_common_word, axis = 1)

print ("original done")

df_all['len_of_query_nouns'] = df_all['search_term_nouns'].map(lambda x:len(str(x).split())).astype(np.int64)
df_all['word_in_title_nouns'] = df_all[['search_term_nouns','product_title_nouns']].apply(str_common_word, axis = 1)
df_all['word_in_description_nouns'] =  df_all[['search_term_nouns','product_description_nouns']].apply(str_common_word, axis = 1)

print ("nouns done")
df_all['len_of_query_stripped'] = df_all['search_term_stripped'].map(lambda x:len(str(x).split())).astype(np.int64)

df_all['word_in_title_stripped'] = df_all[['search_term_stripped','product_title_stripped']].apply(str_common_word, axis = 1)
df_all['word_in_description_stripped'] =  df_all[['search_term_stripped','product_description_stripped']].apply(str_common_word, axis = 1)

df_all.to_csv(sys.argv[1],index=False)


