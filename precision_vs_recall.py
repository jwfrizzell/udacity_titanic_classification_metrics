# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train) #Fit the training data for the decision tree.
predict_dtc = clf1.predict(X_test)
recall_dtc = recall(y_test, predict_dtc)
precision_dtc = precision(y_test, predict_dtc)
print("\nDecision Tree recall: {:.2f} and precision: {:.2f}".format(recall_dtc, 
	precision_dtc))

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
predict_gnb = clf2.predict(X_test)
recall_gnb = recall(y_test, predict_gnb)
precision_gnb = precision(y_test, predict_gnb)
print("\nGaussianNB recall: {:.2f} and precision: {:.2f}".format(recall_gnb, 
	precision_gnb))

results = {
  "Naive Bayes Recall": recall_gnb,
  "Naive Bayes Precision": precision_gnb,
  "Decision Tree Recall": recall_dtc,
  "Decision Tree Precision": precision_dtc
}