import pandas as pd
import numpy as np

def test_data(factor):
	train = pd.read_csv("data/train.csv")
	label = train['label']
	train = train.drop(['label'], 1)
	features = pd.read_csv("data/features_train.csv")
	msk = np.random.rand(len(train)) < factor
	s_train = train[msk]
	s_test = train[~msk]
	s_train.to_csv("data/sample_train.csv", index = False)
	s_test.to_csv("data/sample_test.csv", index = False)

	s_features_train = features[msk]
	s_features_test = features[~msk]

	s_features_train.to_csv("data/sampled_features_train.csv", index = False)
	s_features_test.to_csv("data/sampled_features_test.csv", index = False)

	s_label_train = label[msk]
	s_label_test = label[~msk]

	s_label_train.to_csv("data/label_train.csv", index = False)
	s_label_test.to_csv("data/label_test.csv", index = False)
	return s_train, s_test, s_features_train, s_features_test, s_label_train, s_label_test


def prod_data():
	s_train = pd.read_csv("data/train.csv")
	s_test = pd.read_csv("data/test.csv")
	s_label_train = s_train['label']
	s_train = s_train.drop(['label'], 1)
	s_features_train = pd.read_csv("data/features_train.csv")
	s_features_test = pd.read_csv("data/features_test.csv")
	return s_train, s_test, s_features_train, s_features_test, s_label_train


