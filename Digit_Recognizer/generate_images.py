import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import draw_image
train = pd.read_csv("data/train.csv")
label = train['label']
train = train.drop(['label'], 1)

for i in range(train.shape[0]):
	t = train[i:(i+1)].values
	t = t[0,:]
	draw_image.draw(t,'data/' + str(i) + '_' + str(label[i]) + '.png')




