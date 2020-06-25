
from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 
import pandas as pd 
from numpy.core.umath_tests import inner1d
import time 
#warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn import model_selection  
# load the data 
# read data
data=pd.read_csv("cancer.csv")
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# Independent & dependent variable
X=data.drop(['diagnosis'], axis=1)
y=data['diagnosis']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=40)

def bag__():  
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
	results = model_selection.cross_val_score(model, X, y, cv =kfold) 
	print("Accuracy with Bagging :") 
	print("===================================================================")


	print(results.mean()*100) 
	b=results.mean()*100
	t=end-start
	return b,t
	print("===================================================================")
