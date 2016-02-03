import numpy as np
import pandas as pd

#Loading data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train_y = df_train['relevance']

df_train_id = df_train['id']
df_train_y = pd.DataFrame(df_train_y.values,columns=['relevance'])
df_train_id = pd.DataFrame(df_train_id.values,columns=['id'])
df_train_x = df_train.drop(['id','relevance'], axis=1)
df_test_y = df_test['relevance']
df_test_id = df_test['id'].map(int)
df_test_y = pd.DataFrame(df_test_y.values,columns=['relevance'])
df_test_id = pd.DataFrame(df_test_id.values,columns=['id'])
df_test_x = df_test.drop(['id','relevance'], axis=1)

#saving results
df_train_x.to_csv('train_x.csv',index=False)
df_train_y.to_csv('train_y.csv',index=False)
df_train_id.to_csv('train_id.csv',index=False)
df_test_x.to_csv('test_x.csv',index=False)
df_test_y.to_csv('test_y.csv',index=False)
df_test_id.to_csv('test_id.csv',index=False)

