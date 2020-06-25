# Load libraries
from sklearn.ensemble import AdaBoostClassifier
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
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc

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

def adb__():
	print("===================================================================")
	print("----AdaBoosting----")
	# Create adaboost classifer object
	abc = AdaBoostClassifier(n_estimators=50,
		                 learning_rate=1)
	start=time.time()
	# Train Adaboost Classifer
	model = abc.fit(X_train, y_train)
	end=time.time()
	#Predict the response for test dataset
	y_pred = model.predict(X_test)

	# Model Accuracy, how often is the classifier correct?
	print("Accuracy with adaboost:",metrics.accuracy_score(y_test, y_pred)*100)
	p=metrics.accuracy_score(y_test, y_pred)*100
	t=end-start
	return p,t
	print("===================================================================")

