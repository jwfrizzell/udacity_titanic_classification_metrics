# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
	random_state=1)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train,y_train) #Fit the training set to the  label train set. 
#Make a prediction from the test data and create a confusion matrix showing
#the predictions made. 
#[TP, FP],
#[FN, TN]
confusion_matrix_prediction_dtf = confusion_matrix(y_test, clf1.predict(X_test))
print("Confusion matrix for this Decision Tree:\n", confusion_matrix_prediction_dtf)

clf2 = GaussianNB()
clf2.fit(X_train,y_train)#Fit the training set to the  label train set. 
#Make a prediction from the test data and create a confusion matrix showing
#the predictions made. 
#[TP, FP],
#[FN, TN]
confusion_matrix_prediction_gnb = confusion_matrix(y_test, clf2.predict(X_test))
print("GaussianNB confusion matrix:\n", confusion_matrix_prediction_gnb)

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": confusion_matrix_prediction_gnb,
 "Decision Tree": confusion_matrix_prediction_dtf
}