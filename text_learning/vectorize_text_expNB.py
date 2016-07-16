#!/usr/bin/python

import os
import pickle
import re
import sys
import time

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from email_preprocess import preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, f_classif

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
#temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        #if temp_counter < 200:
        path = os.path.join('..', path[:-1])
        print path
        email = open(path, "r")

        ### use parseOutText to extract the text from the opened email
        text = parseOutText(email)
        text = text.lower()
        ### use str.replace() to remove any instances of the words
        words =  ["sara", "shackleton", "chris", "germani"]
        pattern = re.compile('|'.join(words))
        text = pattern.sub('',text)
        ### append the text to word_data
        word_data.append(text)
        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == "sara":
            from_data.append(0)
        else:
            from_data.append(1)

        email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )

### Load pickle data
#words = pickle.load(open("your_word_data.pkl", "rb"))
#authors = pickle.load(open("your_email_authors.pkl", "rb"))


###-----------------own version of preprocessing----------------
print "===========Own Version of Preprocessing====================="
### in Part 4, do TfIdf vectorization here
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
data = vectorizer.fit_transform(word_data).toarray()
print "emails TfIdf vectorized"

### Split data into training set and testing set
trainX, testX, trainY, testY = train_test_split(data, from_data, test_size=0.1, random_state=42)

### Train Naive Bayes Classifier
classifier = GaussianNB()
t0 = time.time()
classifier.fit(trainX, trainY)
print "training time: ", round(time.time() - t0, 3), "s"
print "classifier trained"
t1 = time.time()
train_pred = classifier.predict(trainX)
print "predict time: ", round(time.time() - t1, 3), "s"
#print "Number of mislabled points out of a total %d points: %d" \
%(len(from_data), (trainY != train_pred).sum())

### Predict test set
t2 = time.time()
test_pred = classifier.predict(testX)
print "test set predict time: ", round(time.time()-t2, 3), "s"
accurracy = classifier.score(testX, testY)
print "accuracy calculated using classifier.score on test set: %f " %(accurracy)
acc = accuracy_score(test_pred, testY)
print "accuracy calculated using sklearn.metrics on test set: %f " % acc

### ------------------- Use Provided email_preprocess.py ------------------------
print "===================Predefined Preprocessing======================="
### TfIdf vectorization
print "vectorize emails"
train_feature, test_feature, train_label, test_label = preprocess("your_word_data.pkl", "your_email_authors.pkl")

### Train classifier
clf = GaussianNB()
tt0 = time.time()
clf.fit(train_feature, train_label)
print "training time: ", round(time.time() - tt0, 3), "s"
print "clf trained"

### Predict test set
tt1 = time.time()
test_pred2 = clf.predict(test_feature)
print "predicting test set time: ", round(time.time() - tt1, 2), "s"
acc_score = clf.score(test_feature, test_label)
print "accuracy: %f" % acc_score
print accuracy_score(test_pred2, test_label)

