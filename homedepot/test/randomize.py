import numpy as np
import pandas as pd
from numpy import random
import sys

df_all = pd.read_csv(sys.argv[1], encoding="ISO-8859-1")
psize = int(sys.argv[4])
size = df_all.shape[0]
trainset = []
testset = []
for i in range(0, size):
    x = random.randint(1,100)
    if x < psize :
       trainset += [i]
    else:
       testset += [i]



df_train = df_all.values[trainset]
df_test = df_all.values[testset]

df_train = pd.DataFrame(df_train, columns = df_all.columns.values)
df_test = pd.DataFrame(df_test, columns = df_all.columns.values)


df_train.to_csv(sys.argv[2],index = False)
df_test.to_csv(sys.argv[3],index = False)

