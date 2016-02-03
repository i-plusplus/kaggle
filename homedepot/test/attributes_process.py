import pandas as pd
import numpy as np
import math
import re

non_decimal = re.compile(r'[^\d.]+')

train = pd.read_csv("train_x.csv", encoding="ISO-8859-1")
test = pd.read_csv("test_x.csv", encoding="ISO-8859-1")

df_all = pd.concat((train,test), axis = 0)

attributes = pd.read_csv("../attributes.csv", encoding="ISO-8859-1")
att = {}
attributes = attributes.values
for i in range(0, len(attributes)):
    if math.isnan(attributes[i][0]) == False:
        att[str(int(attributes[i][0]))] = {}

for i in range(0, len(attributes)):
    if math.isnan(attributes[i][0]) == False:
       k = att[str(int(attributes[i][0]))]
       k[attributes[i][1]] = attributes[i][2]

#number of works in str1 which are present in str2

def str_common_word(str1, str2):
    x = [x for x in str(str1).lower().split() if str(str2).lower().find(x)>=0 ]
    return len(x)

product_index = {}
wh = ['product width (in.)','product height (in.)','product depth (in.)','product weight (lb.)','assembled height (in.)','assembled width (in.)' ,'assembled depth (in.)', 'product length (in.)']

for i in range(0,len(wh)):
    product_index[wh[i]] = i


def match(str1):
    sum2 = [0]*(len(wh)+3)
    query = str(str1[0])
    try:
       product_id = str(str1[1])
       product = att[str(product_id)]
       count = 0
       for key in product:
          value = str(product[key])
          mul = 1
          if key.lower() in wh:
              index = product_index[key.lower()]
              v = str(non_decimal.sub(" ", value))
              q = str(non_decimal.sub(" ",query))
              if query.lower().find(" ft ") >0 or query.lower().find(" ft.") > 0 or query.lower().find(" feet ")> 0:
                   v = str(non_decimal.sub(" ", value).split(" ")[0])
                   v = str(int(float(v)/12))
          elif key.lower() == 'indoor/outdoor':
              index = len(wh)
              v = value.replace("/"," ")
              q = query
          elif key.lower().find('color') > 0:
              index = len(wh)+1
              mul = .5 
              v = value.replace("/"," ")
              q = query.replace("/"," ")
          else:
              index = len(wh)+2
              v = value
              q = query
          l = len(value.split())
          v = str(v)
          q = str(q)
          has = str_common_word(v, q)
          sum2[index] +=  (has*1.0/l)
          count= count+1
       return sum2
    except KeyError:
       return sum2



columns = []
columns += wh
columns += ['indoor/outdoor','color','misc'] 
#columns = ['misc']
new_features = df_all[['search_term','product_uid']].apply(match, axis = 1)
a = np.empty((len(new_features.values), len(new_features.values[0])))
for i in range(0, len(new_features.values)):
    for j in range(0, len(new_features.values[0])):
        a[i][j] = new_features.values[i][j]


new_features = pd.DataFrame(a, columns = columns)
for i in new_features.columns:
    df_all[i] = new_features[i]

train = pd.DataFrame(df_all.values[:train.shape[0]], columns = df_all.columns.values)
test = pd.DataFrame(df_all.values[train.shape[0]:], columns = df_all.columns.values)

#train2 = pd.read_csv("train_combined.csv", encoding="ISO-8859-1")
#test2 = pd.read_csv("test_combined.csv", encoding="ISO-8859-1")

#train2 = pd.concat((train2, train), axis = 1)

#test2 = pd.concat((test2, test), axis = 1)

train.to_csv("train_x.csv", index = False)
test.to_csv("test_x.csv", index = False)




