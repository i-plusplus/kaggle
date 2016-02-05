import pandas as pd
import numpy as np
import math
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag

non_decimal = re.compile(r'[^\d.]+')
train = pd.read_csv("train_x.csv", encoding="ISO-8859-1")
test = pd.read_csv("test_x.csv", encoding="ISO-8859-1")
df_all = pd.concat((train,test), axis = 0)

def str_common_word(str1, str2):
    x = [x for x in str(str1).lower().split() if str(str2).lower().find(x)>=0 ]
    return len(x)

def checkdigits(str1):
    query = str(non_decimal.sub(" ", str1[0]))
    value = str(non_decimal.sub(" ", str1[1]))
    query2 = " ".join([x.split(".")[0] for x in query.split()])
    value2 = " ".join([x.split(".")[0] for x in value.split()])
    return [str_common_word(query,value),str_common_word(query2,value2)]


df_all[['product_title_digits', 'product_title_digits2']] = df_all[['search_term','product_title']].apply(checkdigits, axis = 1)
df_all[['product_description_digits','product_description_digits2']] = df_all[['search_term','product_description']].apply(checkdigits, axis = 1)


def checkdigits(str1):
    query = str(non_decimal.sub(" ", str1[0]))
    value = str(non_decimal.sub(" ", str1[1]))
    return str_common_word(query,value)



train = pd.DataFrame(df_all.values[:train.shape[0]], columns = df_all.columns.values)
test = pd.DataFrame(df_all.values[train.shape[0]:], columns = df_all.columns.values)
train.to_csv("train_x.csv", index = False)
test.to_csv("test_x.csv", index = False)

