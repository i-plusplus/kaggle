import pandas as pd
import numpy as np
import math
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag

stop = stopwords.words('english')
non_decimal = re.compile(r'[^\d.]+')
train = pd.read_csv("train_x.csv", encoding="ISO-8859-1")
test = pd.read_csv("test_x.csv", encoding="ISO-8859-1")
df_all = pd.concat((train,test), axis = 0)

colors = ['black','white']

def common_color(str1):
    query = str(str1[0]).lower().split()
    value = str(str1[1]).lower().split()
    query = [x for x in query if x in colors]
    value = [x for x in value if x in colors]
    return len([x for x in query if x in value])


def str_common_word(str1, str2):
    x = [x for x in str(str1).lower().split() if x in str(str2).lower().split() ]
    return len(x)

def str_common_word2(str1):
    q = [x for x in str(str1[0]).lower().split() if x not in stop]
    v = str(str1[1])
    count = 0
    for i in range(1, len(q)):
        p = " " + q[i-1] + " " + q[i] + " "
        if v.lower().find(p) >= 0:
            count = count + 1
    return count


def checkdigits(str1):
    query = str(non_decimal.sub(" ", str1[0]))
    value = str(non_decimal.sub(" ", str1[1]))
    query2 = " ".join([x.split(".")[0] for x in query.split()])
    value2 = " ".join([x.split(".")[0] for x in value.split()])
    return [str_common_word(query,value),str_common_word(query2,value2)]

def checkdigits2(str1):
    query = str(non_decimal.sub(" ", str1[0]))
    query2 = " ".join([x.split(".")[0] for x in query.split()])
    if len(query.split()) > 0:
       return len(query.split())
    return 0


df_all[['product_title_digits', 'product_title_digits2']] = df_all[['search_term','product_title']].apply(checkdigits, axis = 1)
df_all[['product_description_digits','product_description_digits2']] = df_all[['search_term','product_description']].apply(checkdigits, axis = 1)

df_all['search_term_digit_count'] = df_all[['search_term']].apply(checkdigits2, axis = 1)
df_all['product_title_count'] = df_all[['product_title']].apply(checkdigits2, axis = 1)
df_all['product_description_count'] = df_all[['product_description']].apply(checkdigits2, axis = 1)
df_all['cc'] = df_all[['product_description']].apply(checkdigits2, axis = 1)


#df_all['product_title_color'] = df_all[['search_term','product_title']].apply(str_common_word2, axis = 1)
#df_all['product_description_color'] = df_all[['search_term','product_title']].apply(str_common_word2, axis = 1)

train = pd.DataFrame(df_all.values[:train.shape[0]], columns = df_all.columns.values)
test = pd.DataFrame(df_all.values[train.shape[0]:], columns = df_all.columns.values)
train.to_csv("train_x.csv", index = False)
test.to_csv("test_x.csv", index = False)

