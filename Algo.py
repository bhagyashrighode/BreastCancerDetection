# data cleaning and manipulation 
import pandas as pd
import numpy as np
import time
# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import confusion_matrix

# data Univariate Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# read data
data=pd.read_csv("cancer.csv")

# convert M and B to number form 
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# Independent & dependent variable
X=data.drop(['diagnosis'], axis=1)
y=data['diagnosis']

# Univariate Feature Selection
X_new= SelectKBest(chi2, k=16).fit_transform(X, y)

#StandardScaler
scaler = StandardScaler().fit(X_new)
snX= scaler.transform(X_new)

# Training and spliting data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(snX,y,test_size=0.25,random_state=40)

def KNN__():
	print("===================================================================")
	from sklearn.neighbors import KNeighborsClassifier
	classifier=KNeighborsClassifier(n_neighbors=2)
	start=time.time()
	classifier.fit(X_train,y_train)
	end=time.time()
	y_pred=classifier.predict(X_test)
	from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
	print accuracy_score(y_test,y_pred)*100
	print confusion_matrix(y_test,y_pred)
	print("----KNN----")
	print"Accuracy score using polynomial KNN:",accuracy_score(y_test,y_pred)*100
	print("===================================================================")
	k=accuracy_score(y_test,y_pred)*100
	t=end-start
	return k,t
	
def log_reg__():
	# Logestic Regression
	from sklearn.linear_model import LogisticRegression
	classifier=LogisticRegression(random_state=0)
	start=time.time()
	classifier.fit(X_train,y_train)
	end=time.time()
	y_pred=classifier.predict(X_test)
	#print"Predited data:",y_pred
	#print"Test data:",y_test
	# making confusion metrices
	from sklearn.metrics import confusion_matrix
	cm=confusion_matrix(y_test,y_pred)
	print("===================================================================")
	print("----Logestic Regression----")
	print 'confusion metrics For Logestic Regression:\n',cm
	print 'Accuracy_score using logestic regression:',(metrics.accuracy_score(y_test,classifier.predict(X_test)))*100
	l=(metrics.accuracy_score(y_test,classifier.predict(X_test)))*100
	t=end-start
	return l,t


def Nn__():
	
	#For Nural network
	from sklearn.neural_network import MLPClassifier
	# Neural Network
	mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
	start=time.time()
	mlp.fit(X_train,y_train)
	end=time.time()
	pred=mlp.predict(X_test)

	from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
	print("===================================================================")
	print("----Neural Network----")
	print'Confusion Matrix for Neural Network:\n',confusion_matrix(y_test,pred)
	print 'Classification Report For Neural Network:\n', classification_report(y_test,pred)
	print 'Accuracy using Neural Network:',accuracy_score(y_test,pred)*100
	print("===================================================================")
	n_=accuracy_score(y_test,pred)*100
	t=end-start
	return n_,t
	
def SVM__():
	#SVM
	from sklearn.svm import SVC
	classifier=SVC(kernel='linear',random_state=0)
	start=time.time()	
	classifier.fit(X_train,y_train)
	end=time.time()
	y_pred=classifier.predict(X_test)
	y_pred
	print("===================================================================")
	print("----SVM----")
	from sklearn.metrics import confusion_matrix,classification_report
	c=confusion_matrix(y_test,y_pred)
	print"SVM Classification Report:",(classification_report(y_test,y_pred))
	print 'Accuracy using SVM:',accuracy_score(y_test,y_pred)*100
	print("===================================================================")
	s_=accuracy_score(y_test,y_pred)*100
	t=end-start
	return s_,t
def dt__():
	#Decision tree
	print("===================================================================")
	print("----Decision Tree Classifier----")
	from sklearn.tree import DecisionTreeClassifier
	classifier=DecisionTreeClassifier()
	start=time.time()
	classifier.fit(X_train,y_train)
	end=time.time()
	y_pred=classifier.predict(X_test)
	from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
	print confusion_matrix(y_test,y_pred)
	print classification_report(y_test,y_pred)
	print "Accuracy using DecisionTreeClassifier:",accuracy_score(y_test,y_pred)*100
	print("===================================================================")
	d_=accuracy_score(y_test,y_pred)*100
	t=end-start
	return d_,t
def rft__():
	print("===================================================================")
	print("----Random forest Algorithm----")
	from sklearn.ensemble import IsolationForest
	from sklearn.ensemble import RandomForestClassifier
	regressor=RandomForestClassifier(n_estimators=1000,random_state=0)
	start=time.time()	
	regressor.fit(X_train,y_train)
	end=time.time()
	y_pred=regressor.predict(X_test)
	print "Accuracy using RandomForestClassifier:",accuracy_score(y_test,y_pred)*100
	r_=accuracy_score(y_test,y_pred)*100
	t=end-start
	return r_,t





