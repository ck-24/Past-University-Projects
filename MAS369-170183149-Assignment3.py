
# coding: utf-8

# In[71]:


'''Import packages and data'''

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

songs = pd.read_csv('C:/Users/charl/OneDrive/Documents/Machine Learning/Project 2/songstrain.csv')


'''Logistic Regression'''

X = songs.iloc[:,0:10]
y = songs.song_pop
logreg=LogisticRegression() 
logreg.fit(X,y)
y_pred=logreg.predict(X)
score = logreg.score(X,y)
print(score)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
logreg=LogisticRegression() 
logreg.fit(X_train,y_train) 
y_pred=logreg.predict(X_test)
cm=metrics.confusion_matrix(y_test,y_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)
plt.figure(figsize=(6,6)) 
sns.heatmap(cm,annot=True,fmt="d",linewidths=.5,square=True,cmap="Blues_r") 
plt.ylabel("Actual popularity") 
plt.xlabel("Predicted popularity")
plt.show()
y_pred_proba=logreg.predict_proba(X_test)[::,1] 
fpr,tpr,_=metrics.roc_curve(y_test,y_pred_proba) 
auc=metrics.roc_auc_score(y_test,y_pred_proba) 
print(auc)
plt.plot(fpr,tpr,label="auc="+str(auc)) 
plt.legend(loc=4)
plt.show()

'''Discriminant Analysis'''

X = X.values
y = y.values
lda=LDA() 
lda.fit(X,y)
lda.scalings_ 
lda.explained_variance_ratio_
y_pred=lda.fit(X,y).predict(X)
print(lda.score(X,y))
cm=metrics.confusion_matrix(y,y_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
y_test_pred=lda.fit(X_train,y_train).predict(X_test)
y_predprob=lda.predict_proba(X_test)[::,1]
cm=metrics.confusion_matrix(y_test,y_test_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)
plt.figure(figsize=(6,6)) 
sns.heatmap(cm,annot=True,fmt="d",linewidths=.5,square=True,cmap="Blues_r") 
plt.ylabel("Actual popularity") 
plt.xlabel("Predicted popularity")
plt.show()
fpr,tpr,_=metrics.roc_curve(y_test,y_predprob) 
auc=metrics.roc_auc_score(y_test,y_predprob) 
print(auc)
plt.plot(fpr,tpr,label="auc="+str(auc)) 
plt.legend(loc=4)
plt.show()

'''Decision trees'''

X = songs.iloc[:,0:10]
y = songs.song_pop
treereg=DecisionTreeRegressor(max_depth=2) 
treereg.fit(X,y)
treereg.feature_importances_
features = ["duration", "fade_in", "fade_out", "loudness", "mode", "tempo", "time_sig", "year", "artist_fam","artist_pop"]
export_graphviz(treereg,out_file="tree.dot",feature_names=features)
rfreg=RandomForestRegressor(n_estimators=50) 
rfreg.fit(X,y)
yhat=treereg.predict(X) 
yhat
print(np.sqrt(metrics.mean_squared_error(yhat,y)))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
rfreg.fit(X_train,y_train) 
yhat=rfreg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(yhat,y_test)))
y1=y
rfclf=RandomForestClassifier(n_estimators=50) 
rfclf.fit(X,y1)
yhat1=rfclf.predict(X)
cm=metrics.confusion_matrix(y1,yhat1)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)
X_train,X_test,y1_train,y1_test=train_test_split(X,y1,test_size=0.25)
rfclf.fit(X_train,y1_train)
yhat1=rfclf.predict_proba(X_test)[::,1]


'''Neural Networks'''

X = X.values
y = y.values
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(3, 2), random_state=1, solver='lbfgs')
y_pred = clf.predict(X)
cm=metrics.confusion_matrix(y,y_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
y_pred = clf.predict(X_test)
cm=metrics.confusion_matrix(y_test,y_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)
y_predprob=clf.predict_proba(X_test)[::,1]
auc=metrics.roc_auc_score(y_test,y_predprob) 
print(auc)
plt.plot(fpr,tpr,label="auc="+str(auc)) 
plt.legend(loc=4)
plt.show()
plt.figure(figsize=(6,6)) 
sns.heatmap(cm,annot=True,fmt="d",linewidths=.5,square=True,cmap="Blues_r") 
plt.ylabel("Actual popularity") 
plt.xlabel("Predicted popularity")
plt.show()




