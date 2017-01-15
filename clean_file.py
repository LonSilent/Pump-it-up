import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics, linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import math
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.ensemble import RandomForestClassifier
train_data = pandas.read_csv("./train.csv")
test_data = pandas.read_csv("../new_test.csv")
attributes = ['rows','label','amount_tsh','date_recorded','funder','gps_height','installer','longitude','latitude','wpt_name','num_private','basin','subvillage','region','region_code','district_code','lga','ward','population','public_meeting','recorded_by','scheme_management','scheme_name','permit','construction_year','extraction_type','extraction_type_group','extraction_type_class','management','management_group','payment','payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class','waterpoint_type','waterpoint_type_group']
# 0 nope 1 number 2 todiict 3 year minor
attributesUsage = [0,0,1,0,1,1,
		1,1,1,1,1,
		2,2,2,2,2,
		2,2,1,2,2,
		2,2,2,3,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2]

# attributesUsage = [0,0,1,0,0,1,
		# 0,1,1,0,0,
		# 2,0,2,0,0,
		# 0,0,1,0,0,
		# 0,0,2,3,2,
		# 2,2,0,2,2,
		# 0,2,2,0,0,
		# 2,0,2,2,0]


def toname(type):
	if type == 0:
		return 'non functional'
	elif type == 1:
		return 'functional'
	else:
		return 'functional needs repair'

Ys = train_data.label.tolist()
train_data = pandas.read_csv("../new_train.csv")
Xs = train_data.to_records()
Test_Xs = test_data.to_records()
ids = test_data.id.tolist()
Xs_in_dict = []

print('Transforming Training Data...')
for x in range(0,len(Xs)):
	if x%10000 == 0:
		print(x)
	obj = {}
	for y in range(0,len(attributesUsage)):
		if type(Xs[x][y])!= str and math.isnan(Xs[x][y]):
			Xs[x][y] = -1
		if attributesUsage[y] == 1 or attributesUsage[y] == 2:
			obj[attributes[y]] = Xs[x][y]
		elif attributesUsage[y] == 3:
			obj[attributes[y]] = 2016 - Xs[x][y]
	Xs_in_dict.append(obj)

print('DictVectorizering...')
DictVectorizer = DictVectorizer()
Xs = DictVectorizer.fit_transform(Xs_in_dict).toarray()
print(len(Xs[0]))
# selector = SelectKBest(k=100)
# Xs = selector.fit_transform(Xs,Ys)
# print(len(Xs[0]))
print('Training...')
# model = linear_model.LogisticRegression()
# model = AdaBoostClassifier();
model = RandomForestClassifier(n_estimators=300,n_jobs=12)
# model = IsolationForest(n_estimators=200, n_jobs=12)
# model = GaussianNB();
# model = linear_model.LinearRegression()
# model = tree.DecisionTreeClassifier()
model.fit(Xs,Ys)

print('Estimating...')
preds = model.predict(Xs)
print(preds)
print(metrics.accuracy_score(Ys,preds))

print('Transforming Testing Data...')
Xs_in_dict = []
for x in range(0,len(Test_Xs)):
	if x%10000 == 0:
		print(x)
	obj = {}
	for y in range(0,len(attributesUsage)):
		if type(Test_Xs[x][y])!= str and math.isnan(Test_Xs[x][y]):
			Test_Xs[x][y] = -1
		if attributesUsage[y] == 1 or attributesUsage[y] == 2:
			obj[attributes[y]] = Test_Xs[x][y]
		elif attributesUsage[y] == 3:
			obj[attributes[y]] = 2016 - Test_Xs[x][y]
	Xs_in_dict.append(obj)
Test_Xs = DictVectorizer.transform(Xs_in_dict).toarray()

print('Predicting...')
preds_test = model.predict(Test_Xs)
SavingContent = "";
for x in range(0,len(ids)):
	SavingContent = SavingContent + str(ids[x]) + "," + toname(preds_test[x]) + "\n"

output_path = './preds_new.csv'
print('saving file...')
savefile = open(output_path,'w')
savefile.write('id,status_group\n')
savefile.write(SavingContent)
savefile.close()

print('done!')