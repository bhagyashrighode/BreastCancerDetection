# Load libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# data cleaning and manipulation 
import pandas as pd
import numpy as np
import time
#warning
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning) 

# read data
data=pd.read_csv("cancer.csv")

# convert M and B to number form 
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# Independent & dependent variable
X=data.drop(['diagnosis'], axis=1)
y=data['diagnosis']

# Training and spliting data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
def gbd__():
	print("===================================================================")
	print("----Gradient Boosting----")
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
	start=time.time()
	clf.fit(X_train, y_train)
	end=time.time()
	clf.score(X_test, y_test)
	print "Accuracy with Gradient Boosting:",clf.score(X_test, y_test)*100
	g=clf.score(X_test, y_test)*100
	t=end-start	
	return g,t
	print("===================================================================")

