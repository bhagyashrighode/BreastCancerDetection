from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data cleaning and manipulation 
import pandas as pd
import numpy as np
import time
# read data
data=pd.read_csv("cancer.csv")

# convert M and B to number form 
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# Independent & dependent variable
X=data.drop(['diagnosis'], axis=1)
y=data['diagnosis']


# Training and spliting data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

def xg__():
	print("===================================================================")
	print("----XGboosting----")
	#fit model on taining data
	model=XGBClassifier()
	start=time.time()
	model.fit(X_train,y_train)
	end=time.time()
	print(model)
	# make predictions
	y_pred=model.predict(X_test)
	predictions=[round(value) for value in y_pred]

	# evaluate accuracy

	acc=accuracy_score(y_test,predictions)
	print('Accuracy with XGBoosting:',(acc*100))
	x=(acc*100)
	t=end-start
	return x,t

	print("===================================================================")

