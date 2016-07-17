#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state = 42)
clf_cv = DecisionTreeClassifier()
clf_cv.fit(features_train, labels_train)
labels_pred = clf_cv.predict(features_test)
print clf_cv.score(features_test, labels_test)
print confusion_matrix(labels_test, labels_pred)
#print sum(labels_test)
#print len(labels_test)
