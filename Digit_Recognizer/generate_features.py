import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
train = pd.read_csv("data/train.csv")
label = train['label']
train = train.drop(['label'], 1)
def get_block_features(df, sectors):
    features = []
    columns = []
    for i in range(0, np.size(sectors)):
        features = features + [[]]
        columns = columns + [[]]
    for s in range(0, np.size(sectors)):
        sector = sectors[s]
        for i in range(0,sector):
            for j in range(0,sector):
                columns[s] = columns[s] + [str(sector) + "_" + str(i) + "_" + str(j)]
    for k in range(df.shape[0]):
        print(k)	
        t = df[k:(k+1)].values
        t = t[0,:]
        t = np.split(t,28)
        t = pd.DataFrame(t)
        for s in range(0,np.size(sectors)):
            sector = sectors[s]
            length = int(np.shape(t)[1]/sector)	
            w = []
            for i in range(0,sector):
                for j in range(0,sector):
                    l = t[i:(i+1)*length][list(range(j,(j+1)*length))]
                    l = l.values.ravel()
                    white = np.size(l[l<=1])
                    black = np.size(l[l>1])
                    w = w + [black]
            w = np.asarray(w)/sum(w)
            features[s] = features[s] + [list(w)]
    result = pd.DataFrame(features[0], columns = columns[0])
    for i in range(1, np.size(sectors)):
        result = pd.concat([result, pd.DataFrame(features[i], columns = columns[i])], axis = 1)
    return result


df = get_block_features(train, [2,4,7])

df.to_csv("data/features_train.csv", index=False)
test = pd.read_csv("data/test.csv")
df = get_block_features(test, [2,4,7])
df.to_csv("data/features_test.csv", index = False)
label.to_csv("data/label.csv", index=False)

