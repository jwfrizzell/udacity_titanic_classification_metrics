# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
	random_state=1)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train) #Fit the training data to the Decision Tree
f1_score_dtc = f1_score(y_test, clf1.predict(X_test))
print "Decision Tree F1 score: {:.2f}".format(f1_score_dtc)

clf2 = GaussianNB()
clf2.fit(X_test, y_test)
f1_score_gnb = f1_score(y_test, clf2.predict(X_test))
print "GaussianNB F1 score: {:.2f}".format(f1_score_gnb)

F1_scores = {
 "Naive Bayes": f1_score_gnb,
 "Decision Tree": f1_score_dtc
}