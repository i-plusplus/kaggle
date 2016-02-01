import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import threading
import datetime
import os

#os.system("/opt/rh/python33/root/usr/bin/python3  splitter.py")

class myThread (threading.Thread):
    def __init__(self, file):
        threading.Thread.__init__(self)
        self.file = file
    def run(self):
        os.system("/opt/rh/python33/root/usr/bin/python3 processor.py " + str(self.file))


threads = {}
t = 32
for i in range(0,t):
    threads[i] = myThread(str(i) + ".csv")
    threads[i].start()

for i in range(0,t):
    threads[i].join()



df_all  = pd.read_csv("0.csv")
for i in range(1,t):
    df_lowers = pd.read_csv(str(i) + ".csv")
    df_all = pd.concat((df_all,df_lowers), axis = 0)

df_all.to_csv("combined.csv", index = False)

