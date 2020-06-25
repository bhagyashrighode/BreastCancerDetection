#GUI Designing
from tkinter import *
import tkinter as tk
import cv2
from PIL import Image,ImageTk 
from tkinter import ttk
from PIL import Image
from resizeimage import resizeimage
import proj as p

def knn():
	pr=KNN__()
	e1.delete(first =0, last =100)	
	e1.insert(1,pr)
def log_reg():
	
	lr=log_reg__()
	e2.delete(first =0, last =100)	
	e2.insert(1,lr)
def nn_():
	n=Nn__()
	e3.delete(first =0, last =100)	
	e3.insert(1,n)

def svm_():
	print n
	e4.delete(first =0, last =100)	
	e4.insert(1,svm)
def dec_t():
	print n
	e5.delete(first =0, last =100)	
	e5.insert(1,dt)
def rm_f():
	print n
	e6.delete(first =0, last =100)	
	e6.insert(1,rt)	

# Applying PCA function on training 
# and testing set of X component 
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
explained_variance = pca.explained_variance_ratio_

#Algoritham with PCA
print("===================================================================")
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print accuracy_score(y_test,y_pred)*100
print confusion_matrix(y_test,y_pred)
print("----KNN----")
print"Accuracy score using polynomial KNN:",accuracy_score(y_test,y_pred)*100
print("===================================================================")
pra=accuracy_score(y_test,y_pred)*100

 
# Logestic Regression with PCA
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
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
lra=(metrics.accuracy_score(y_test,classifier.predict(X_test)))*100

# Neural Network with PCA
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,y_train)
pred=mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("===================================================================")
print("----Neural Network with PCA----")
print'Confusion Matrix for Neural Network:\n',confusion_matrix(y_test,pred)
print 'Classification Report For Neural Network:\n', classification_report(y_test,pred)
print 'Accuracy using Neural Network:',accuracy_score(y_test,pred)*100
print("===================================================================")
na=accuracy_score(y_test,pred)*100

#SVM with PCA
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred
print("===================================================================")
print("----SVM with PCA----")
from sklearn.metrics import confusion_matrix,classification_report
c=confusion_matrix(y_test,y_pred)
print"SVM Classification Report:",(classification_report(y_test,y_pred))
print 'Accuracy using SVM:',accuracy_score(y_test,pred)*100
print("===================================================================")
svma=accuracy_score(y_test,pred)*100

#Decision tree with PCA
print("===================================================================")
print("----Decision Tree Classifier with PCA----")
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print confusion_matrix(y_test,y_pred)
print classification_report(y_test,y_pred)
print "Accuracy using DecisionTreeClassifier:",accuracy_score(y_test,y_pred)*100
print("===================================================================")
dta=accuracy_score(y_test,y_pred)*100

#Random forest with PCA
print("===================================================================")
print("----Random forest Algorithm with PCA----")
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
regressor=RandomForestClassifier(n_estimators=1000,random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
print "Accuracy using RandomForestClassifier:",accuracy_score(y_test,y_pred)*100
rta=accuracy_score(y_test,y_pred)*100

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
  
results = model_selection.cross_val_score(model, X, y, cv = kfold) 
print("Accuracy with Bagging :") 
print(results.mean()*100) 
bg=results.mean()*100

#Ada Boosting
print("===================================================================")
print("----Adaboosting----")
# Training and spliting data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy with adaboost:",metrics.accuracy_score(y_test, y_pred)*100)
ada=metrics.accuracy_score(y_test, y_pred)*100

#gradient Boosting
print("===================================================================")
print("----Gradient Boosting----")
# Training and spliting data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print "Accuracy with Gradient Boosting:",clf.score(X_test, y_test)*100
gd=clf.score(X_test, y_test)*100

#XGBoosting
print("===================================================================")
print("----XGBoosting----")
# Training and spliting data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
#fit model on taining data
model=XGBClassifier()
model.fit(X_train,y_train)
print(model)
# make predictions
y_pred=model.predict(X_test)
predictions=[round(value) for value in y_pred]
# evaluate accuracy
acc=accuracy_score(y_test,predictions)
print('Accuracy with XGBoosting:',(acc*100))
xg=(acc*100)
print("===================================================================")
def knna():
	print pr
	e7.delete(first =0, last =100)	
	e7.insert(1,pra)
def log_rega():
	
	print lr
	e8.delete(first =0, last =100)	
	e8.insert(1,lra)
def nn_a():
	print n
	e9.delete(first =0, last =100)	
	e9.insert(1,na)

def svm_a():
	print n
	e10.delete(first =0, last =100)	
	e10.insert(1,svma)
def dec_ta():
	print n
	e11.delete(first =0, last =100)	
	e11.insert(1,dta)
def bg_():
	e13.delete(first =0, last =100)	
	e13.insert(1,bg)
def rm_fa():
	print n
	e12.delete(first =0, last =100)	
	e12.insert(1,rta)	
def adb_():
	print n
	e14.delete(first =0, last =100)	
	e14.insert(1,ada)
def gdb_():
	print n
	e15.delete(first =0, last =100)	
	e15.insert(1,gd)
def xg_():
	print n
	e16.delete(first =0, last =100)	
	e16.insert(1,xg)

root = Tk()
root.title(string='Breast Cancer Detection Algoritham and accuracy')
root.configure(background='cyan')
root.geometry('1000x800')

#Image Displaying on gui
canvas = Canvas(root, width = 200, height = 100,bg='cyan')  
canvas.grid(row=0,column=1)  
img = ImageTk.PhotoImage(Image.open('cancer.png'))  
canvas.create_image(0,0,anchor=tk.NW,image=img)

canvas1= Canvas(root, width = 200, height = 200,bg='cyan')  
canvas1.grid(row=11,column=1)  
img1= ImageTk.PhotoImage(Image.open('cancer1.jpeg'))  
canvas1.create_image(0,0,anchor=tk.NW,image=img1)

canvas2= Canvas(root, width = 200, height = 200,bg='cyan')  
canvas2.grid(row=11,column=2)  
img2= ImageTk.PhotoImage(Image.open('cancer3.png'))  
canvas2.create_image(0,0,anchor=tk.NW,image=img2)

#Label,Button,Entry displaying on GUI without PCA
Label(root,text="Alogrithams",font=('Times',18),fg='grey',bg='cyan').grid(row=1,column=0)
Label(root,text="Accuracy",font=('Times',18),fg='grey',bg='cyan').grid(row=1,column=1)
e1 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e1.grid(row=2, column=1) 
b1= Button(root, text='   KNN   ', bg='grey', fg='white', font='Times 18',width=20,command=knn).grid(row=2, column=0)
e2 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e2.grid(row=3, column=1) 
b2= Button(root, text='  Logestic  Regression  ', bg='grey', fg='white', font='Times 18',width=20,command=log_reg).grid(row=3, column=0)
e3 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e3.grid(row=4, column=1) 
b3= Button(root, text='    Neural  Network   ', bg='grey', fg='white', font='Times 18',width=20,command=nn_).grid(row=4, column=0)
e4 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e4.grid(row=5, column=1) 
b4= Button(root, text='         _SVM_       ', bg='grey', fg='white', font='Times 18',width=20,command=svm_).grid(row=5, column=0)
e5 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e5.grid(row=6, column=1) 
b5= Button(root, text='   DecisionTree     ', bg='grey', fg='white', font='Times 18',width=20,command=dec_t).grid(row=6, column=0)
e6 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e6.grid(row=7, column=1) 
b6= Button(root, text=' Random Forest ', bg='grey', fg='white', font='Times 18',width=20,command=rm_f).grid(row=7, column=0)

#Label,Button,Entry displaying on GUI with PCA
Label(root,text="Alogrithams with PCA",font=('Times',18),fg='grey',bg='cyan').grid(row=1,column=2)
Label(root,text="   Accuracy with PCA",font=('Times',18),fg='grey',bg='cyan').grid(row=1,column=3)
e7 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e7.grid(row=2, column=3) 
b7= Button(root, text='  KNN  ', bg='grey', fg='white', font='Times 18',width=20,command=knna).grid(row=2, column=2)
e8 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e8.grid(row=3, column=3) 
b8= Button(root, text='  Logestic  Regression  ', bg='grey', fg='white', font='Times 18',width=20,command=log_rega).grid(row=3, column=2)
e9 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e9.grid(row=4, column=3) 
b9= Button(root, text='    Neural  Network   ', bg='grey', fg='white', font='Times 18',width=20,command=nn_a).grid(row=4, column=2)
e10 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e10.grid(row=5, column=3) 
b10= Button(root, text='         _SVM_       ', bg='grey', fg='white', font='Times 18',width=20,command=svm_a).grid(row=5, column=2)
e11 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e11.grid(row=6, column=3) 
b11= Button(root, text='   DecisionTree     ', bg='grey', fg='white', font='Times 18',width=20,command=dec_ta).grid(row=6, column=2)
e12 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e12.grid(row=7, column=3) 
b12= Button(root, text=' Random Forest ', bg='grey', fg='white', font='Times 18',width=20,command=rm_fa).grid(row=7, column=2)

#label,button,entry for Bagging and Boosting
Label(root,text="   Bagging & boosting    ",font=('Times',18),fg='grey',bg='cyan').grid(row=8,column=1)
b13= Button(root, text=' Bagging ', bg='grey', fg='white', font='Times 18',width=20,command=bg_).grid(row=9, column=0)
e13 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e13.grid(row=9, column=1) 

b14= Button(root, text=' AdaBoosting ', bg='grey', fg='white', font='Times 18',width=20,command=adb_).grid(row=9, column=2)
e14 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e14.grid(row=9, column=3) 

b15= Button(root, text=' Gradient Boosting ', bg='grey', fg='white', font='Times 18',width=20,command=gdb_).grid(row=10, column=2)
e15 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e15.grid(row=10, column=3) 

b16= Button(root, text=' XGBoosting ', bg='grey', fg='white', font='Times 18',width=20,command=xg_).grid(row=10, column=0)
e16 = Entry(root,font="Times 18", fg='black', bg='white', width=17) 
e16.grid(row=10, column=1) 


root.mainloop()
