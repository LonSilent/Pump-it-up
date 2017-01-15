import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics, linear_model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import math
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from collections import Counter
import numpy
import csv

# test_data = pandas.read_csv('.data/test_values.csv')

def merge_train_label():
	print("Reading files...")
	train_value = pandas.read_csv('./data/train_values.csv')
	train_label = pandas.read_csv('./data/train_label.csv')

	print("merge two data...")
	train_data = train_value.merge(train_label, how='outer', on='id', sort=True)
	train_data = train_data.fillna(train_data.median())

	train_data.to_csv('./data/train_data.csv',index=False)

def preprocess_label():
	train_label = pandas.read_csv('./data/train_labels.csv')
	column_labels = list(train_label.columns.values)
	column_labels.remove("id")

	for i in column_labels:
		unique_value = train_label[i].unique()
		size = len(unique_value)
		print(size)
		for j in range(size):
			if unique_value[j] != "nan":
				train_label.loc[train_label[i] == unique_value[j], i] = j

	train_label.to_csv("./data/train_label.csv", index=False)

def level_down(attr, infile, outfile):
	print("Reading files...")

	train_value = pandas.read_csv(infile)
	# print(train_value)

	attr_value = train_value[attr].tolist()
	# print(attr_value)
	count = Counter(attr_value).most_common(10)
	print(count)
	reserve = [x for x,y in count if str(x)!='nan']
	print(reserve)

	values = []
	with open(infile) as f:
		obj = {}
		keys = f.readline().strip().split(',')
		print(keys)
		for line in f:
			pattern = line.strip().split(',')
			values.append(pattern)

	idx_attr = keys.index(attr)
	for v in values:
		if v[idx_attr] not in reserve:
			v[idx_attr] = 'Others'

	print(values[:10])

	with open(outfile,'w') as f:
		writer = csv.writer(f)
		writer.writerow(keys)
		for v in values:
			writer.writerow(v)

def median_replace(attr, infile, outfile):
	print("Reading files...")

	train_value = pandas.read_csv(infile)
	attr_value = train_value[attr]
	print(attr_value)
	attr_median = attr_value.median()
	print(attr_value.median())

	values = []
	with open(infile) as f:
		obj = {}
		keys = f.readline().strip().split(',')
		print(keys)
		for line in f:
			pattern = line.strip().split(',')
			values.append(pattern)

	idx_attr = keys.index(attr)
	for v in values:
		if int(v[idx_attr]) == 0:
			v[idx_attr] = attr_median

	with open(outfile,'w') as f:
		writer = csv.writer(f)
		writer.writerow(keys)
		for v in values:
			writer.writerow(v)

if __name__ == '__main__':
	# preprocess_label()
	# merge_train_label()
	level_down("wpt_name",'new_train.csv','new_train.csv')
	level_down("wpt_name",'new_test.csv','new_test.csv')
	# median_replace("population",'new_train.csv','new_train_2.csv')
	# median_replace("population",'new_test.csv','new_test_2.csv')
