#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import NuSVC

### Nearest Neighbors Classifier
###### K Nearest Neighbors
clf_knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
clf_knn.fit(features_train, labels_train)
acc_knn = clf_knn.score(features_test, labels_test)
print "KNN accuracy: ", acc_knn
###### Radius Neighbors Classifier
clf_rdn = RadiusNeighborsClassifier(radius=100.0, weights='distance')
clf_rdn.fit(features_train, labels_train)
acc_rdn = clf_rdn.score(features_test, labels_test)
print "RDN accuracy: ", acc_rdn

### Random Forest Classifier
clf_rdf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=100)
clf_rdf.fit(features_train, labels_train)
acc_rdf = clf_rdf.score(features_test, labels_test)
print "Random Forest accuracy: ", acc_rdf

### Adaptive Boost Classifier
estimator = NuSVC(nu=.5, kernel='rbf', gamma=100)
clf_adb = AdaBoostClassifier(base_estimator=estimator, n_estimators=70,\
                             learning_rate=0.25, algorithm="SAMME")
clf_adb.fit(features_train, labels_train)
acc_adb = clf_adb.score(features_test, labels_test)
print "Adaptive Boost accuracy: ", acc_adb


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass


