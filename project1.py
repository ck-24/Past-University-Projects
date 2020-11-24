# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 00:22:16 2019

@author: charl
"""

'''Import packages and data'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
plt.style.use("seaborn")

all = pd.read_csv('C:/Users/charl/OneDrive/Documents/Machine Learning/winequality-all.csv')
red = pd.read_csv('C:/Users/charl/OneDrive/Documents/Machine Learning/winequality-red.csv')
white = pd.read_csv('C:/Users/charl/OneDrive/Documents/Machine Learning/winequality-white.csv')

'''Clean data'''


'''Check for missing values'''

all.shape
all.describe()

'''Get rid of duplicates'''

all.drop_duplicates(inplace=True)
red.drop_duplicates(inplace=True)
white.drop_duplicates(inplace=True)


'''Find anomalies'''

#print(sns.pairplot(all))
all.loc[all["residual sugar"]>60]
all.loc[all["density"]>1.01]
all.loc[all["free sulfur dioxide"]>200]
all.loc[all["citric acid"]>1.2]
all.loc[all["chlorides"]>0.6]

'''Get rid of anomalies'''

all.drop([2781,1653,4745,745,3152,5049,5156], axis=0, inplace=True)
white.drop([2781,1653,4745,745,3152], axis=0, inplace=True)
white.shape
red.drop([151,258], axis=0, inplace=True)
#all=all.drop(["colour"], axis=1)
categories = all.T.index


'''PCA'''

scaler=StandardScaler() 
scaler.fit(all) 
all_sc=scaler.transform(all)

pca=PCA(n_components=10) 
pca.fit(all_sc)
pca.explained_variance_ratio_
print("***********")
print(np.cumsum(pca.explained_variance_ratio_))
print("***********")
b = pca.components_

PCs = pd.DataFrame()
for a in range (13):
    PCs[categories[a]]=b[:,a]
print(PCs)   
print("***********")
p=pca.fit_transform(all_sc)

plt.scatter(p[0:3955,0],p[0:3955,1],color='black')
plt.scatter(p[3956:5313,0],p[3956:5313,1],color='red')
plt.xlabel("PC1") 
plt.ylabel("PC2")
plt.axis("equal")
plt.show()

plt.xlim(-1,1) 
plt.ylim(-1,1) 
plt.xlabel("PC1") 
plt.ylabel("PC2") 
x=p[:,0] 
y=p[:,1] 
coeff=np.transpose(pca.components_[0:2,:]) 
n=all.shape[1] 
scalex=1.0/(x.max()-x.min()) 
scaley=1.0/(y.max()-y.min()) 
plt.scatter(x*scalex,y*scaley, color='yellow', alpha=0.2) 
for i in range(n): 
    plt.arrow(0,0,coeff[i,0],coeff[i,1],color="black",alpha=0.5) 
    plt.text(coeff[i,0]*1.15,coeff[i,1]*1.15,categories[i],color="black")
plt.show()
print(red.shape)
'''Adding columns'''

all["total acidity"] = all["fixed acidity"] + all["volatile acidity"]
white["total acidity"] = white["fixed acidity"] + white["volatile acidity"]
red["total acidity"] = red["fixed acidity"] + red["volatile acidity"]

s=sns.heatmap(all.corr()) 
s.set_yticklabels(s.get_yticklabels(),rotation=35,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=35,fontsize=7)
plt.show()
print(all.shape)
max(all.quality)
all.loc[all["quality"]>8]

