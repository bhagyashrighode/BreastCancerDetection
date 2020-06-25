# data cleaning and manipulation 
import pandas as pd
import numpy as np
import numpy.core.umath_tests
from numpy.core.umath_tests import inner1d
import time
# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt 

# data Univariate Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# data recursive feature elimination with cross validation
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
#Classifiers
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
#For Nural network
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
def time_():
	# read data
	data=pd.read_csv("cancer.csv")
	# convert M and B to number form 
	data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

	# Independent & dependent variable
	X=data.drop(['diagnosis'], axis=1)
	y=data['diagnosis']

	# Normalization of data
	#nX=preprocessing.normalize(X)
	#ny=preprocessing.normalize(y)

	# Univariate Feature Selection
	X_new= SelectKBest(chi2, k=16).fit_transform(X, y)
	# Applying PCA function on training 
	# and testing set of X component 
	# Training and spliting data 


	#StandardScaler
	scaler = StandardScaler().fit(X_new)
	snX= scaler.transform(X_new)
	from sklearn.model_selection import train_test_split
	X_train,X_test,y_train,y_test=train_test_split(snX,y,test_size=0.25,random_state=40)

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
	t1=end-start
	a=(accuracy_score(y_test,y_pred)*100)
	print("===================================================================")

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
	b=((metrics.accuracy_score(y_test,classifier.predict(X_test)))*100)
	t2=end-start
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
	c1=accuracy_score(y_test,pred)*100
	t3=end-start	
	print("===================================================================")

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
	print 'Accuracy using SVM:',accuracy_score(y_test,pred)*100
	d=accuracy_score(y_test,pred)*100
	t4=end-start
	print("===================================================================")

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
	e=accuracy_score(y_test,y_pred)*100
	t5=end-start	
	print("===================================================================")

	#Random forest
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
	f=accuracy_score(y_test,y_pred)*100
	t6=end-start

	from sklearn.decomposition import PCA 
	pca = PCA(n_components = 2) 
	X_train = pca.fit_transform(X_train) 
	X_test = pca.transform(X_test) 
	explained_variance = pca.explained_variance_ratio_

	#Algoritham with PCA
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
	g=accuracy_score(y_test,y_pred)*100
	t7=end-start	
	print("===================================================================")
	pra=accuracy_score(y_test,y_pred)*100

	 
	# Logestic Regression with PCA
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
	print("----Logestic Regression with PCA----")
	print 'confusion metrics For Logestic Regression:\n',cm
	print 'Accuracy_score using logestic regression:',(metrics.accuracy_score(y_test,classifier.predict(X_test)))*100
	h=(metrics.accuracy_score(y_test,classifier.predict(X_test)))*100
	t8=end-start
	# Neural Network with PCA
	mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
	start=time.time()
	mlp.fit(X_train,y_train)
	end=time.time()
	pred=mlp.predict(X_test)
	from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
	print("===================================================================")
	print("----Neural Network with PCA----")
	print'Confusion Matrix for Neural Network:\n',confusion_matrix(y_test,pred)
	print 'Classification Report For Neural Network:\n', classification_report(y_test,pred)
	print 'Accuracy using Neural Network:',accuracy_score(y_test,pred)*100
	print("===================================================================")
	i=accuracy_score(y_test,pred)*100
	t9=end-start
	#SVM with PCA
	from sklearn.svm import SVC
	classifier=SVC(kernel='linear',random_state=0)
	start=time.time()
	classifier.fit(X_train,y_train)
	end=time.time()
	y_pred=classifier.predict(X_test)
	y_pred
	print("===================================================================")
	print("----SVM with PCA----")
	from sklearn.metrics import confusion_matrix,classification_report
	cm=confusion_matrix(y_test,y_pred)
	print"SVM Classification Report:",(classification_report(y_test,y_pred))
	print 'Accuracy using SVM:',accuracy_score(y_test,pred)*100
	print("===================================================================")
	j=accuracy_score(y_test,pred)*100
	t10=end-start
	#Decision tree with PCA
	print("===================================================================")
	print("----Decision Tree Classifier with PCA----")
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
	k=accuracy_score(y_test,y_pred)*100
	t11=end-start
	#Random forest with PCA
	print("===================================================================")
	print("----Random forest Algorithm with PCA----")
	from sklearn.ensemble import IsolationForest
	from sklearn.ensemble import RandomForestClassifier
	regressor=RandomForestClassifier(n_estimators=1000,random_state=0)
	start=time.time()
	regressor.fit(X_train,y_train)
	end=time.time()
	y_pred=regressor.predict(X_test)
	print "Accuracy using RandomForestClassifier:",accuracy_score(y_test,y_pred)*100
	l=accuracy_score(y_test,y_pred)*100
	t12=end-start
	#bagging
	print("===================================================================")
	print("----Bagging----")
	seed = 0
	kfold = model_selection.KFold(n_splits = 10, 
		               random_state = seed) 
	# initialize the base classifier 
	base_cls = DecisionTreeClassifier() 
	  
	# no. of base classifier 
	num_trees = 500
	  
	# bagging classifier 
	
	model = BaggingClassifier(base_estimator = base_cls, 
		                  n_estimators = num_trees, 
		                  random_state = seed) 
	start=time.time()
	model.fit(X_train,y_train)  
	end=time.time()
	results = model_selection.cross_val_score(model, X, y, cv = kfold) 
	print("Accuracy with Bagging :") 
	print(results.mean()*100) 
	m=results.mean()*100
	t13=end-start
	#Ada Boosting
	print("===================================================================")
	print("----Adaboosting----")
	# Training and spliting data 
	from sklearn.model_selection import train_test_split
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
	# Create adaboost classifer object
	abc= AdaBoostClassifier(n_estimators=50,
		                 learning_rate=1)
	start=time.time()
	# Train Adaboost Classifer
	model = abc.fit(X_train, y_train)
	end=time.time()
	#Predict the response for test dataset
	y_pred = model.predict(X_test)
	# Model Accuracy, how often is the classifier correct?
	print("Accuracy with adaboost:",metrics.accuracy_score(y_test, y_pred)*100)
	o=metrics.accuracy_score(y_test, y_pred)*100
	t14=end-start
	#gradient Boosting
	print("===================================================================")
	print("----Gradient Boosting----")
	# Training and spliting data 
	from sklearn.model_selection import train_test_split
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
	start=time.time()
	clf.fit(X_train, y_train)
	end=time.time()
	clf.score(X_test, y_test)
	print "Accuracy with Gradient Boosting:",clf.score(X_test, y_test)*100
	p=clf.score(X_test, y_test)*100
	t15=end-start
	#XGBoosting
	print("===================================================================")
	print("----XGBoosting----")
	# Training and spliting data 
	from sklearn.model_selection import train_test_split
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
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
	q=(acc*100)
	t16=end-start 
	accuracy_=[a,b,c1,d,e,f,g,h,i,j,k,l,m,o,p,q]
	t=[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16]	
	#print "Accuracy:",accuracy_
	return t

	

