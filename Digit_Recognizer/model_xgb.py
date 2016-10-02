import pandas as pd
import numpy as np
import draw_image as di
import sys
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import dataGatherer as dg
isTest = int(sys.argv[1])
if isTest == 1:
	train, test, feature_train, feature_test, label_train, label_test = dg.test_data(.8)
else:
	train, test, feature_train, feature_test, label_train = dg.prod_data()
f_train = pd.concat([train,feature_train], axis = 1)
f_test = pd.concat([test,feature_test], axis = 1)
xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=50, objective='multi:softprob', subsample=1.0, colsample_bytree=1, seed=0)
le = LabelEncoder()
y = le.fit_transform(label_train.values)

xgb.fit(f_train.values, y)
y_pred = xgb.predict(f_test.values)
y_pred = le.inverse_transform(y_pred)
if isTest == 1 :
	y_f = y_pred == label_test.values
	len(y_f[y_f==False])
	len(y_f[y_f==True])
	t = test[~y_f]
	l = label_test[~y_f]
	l_p = y_pred[~y_f]
	for i in range(0, len(l)):
		di.draw(t[i:i+1].values[0,], "images/prob_" + str(i) + "_" + str(l.values[i]) + "_" + str(l_p[i]) )
else :
	index = range(1,len(y_pred)+1)
	index = pd.DataFrame(index, columns = ['ImageId'])
	y_pred = pd.DataFrame(y_pred, columns = ['Label'])
	final = pd.concat([index, y_pred], axis = 1)
	final.to_csv("data/result.csv", index = False)


