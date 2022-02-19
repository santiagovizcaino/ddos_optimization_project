# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:39:26 2021

@author: santy
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.svm import SVC
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#data = pd.read_csv(r"C:\Users\santy\Desktop\dataset_articulo.csv", encoding = 'utf-8') # acc=1
data = pd.read_csv(r"C:\Users\santy\Desktop\dataset_003.csv", encoding = 'utf-8') # acc=093


#VERIFICAR SEPARACION DE CLASES
# df0=data[data.type=='ddos']
# df1=data[data.type=='normal']

# plt.scatter(df0['ack'], df0['srcport'],color='green',marker='+')
# plt.scatter(df1['ack'], df1['srcport'],color='blue',marker='+')

#CATEGORIZACION DE VARIABLES
data['type'] = pd.factorize(data.type)[0]


#DIVISION DEL DATASET
from sklearn.model_selection import train_test_split

X = data.drop(columns='type')
y= data["type"]
x_train,x_test,y_train,y_test=train_data=train_test_split(data,y.values.reshape(-1,1),test_size=0.3,random_state = 1234,shuffle= True)

#SVM

import seaborn as sns
from sklearn.svm import SVC

#model = SVC(C = 1, kernel = 'linear', random_state=123)
model = SVC() 
model.fit(x_train, y_train)


 

model.score(x_test,y_test)
y_pred=model.predict(x_test)


from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,precision_score,recall_score,f1_score
import seaborn as sn
print ('Accuracy ',accuracy_score(y_test,y_pred))
print ('Preccision ',precision_score(y_test,y_pred))
print ('Recall ',recall_score(y_test,y_pred))
print ('F1 ',f1_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sn.heatmap(cm, annot=labels, fmt='',cmap='YlGn')

data_corelacion= data.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(data_corelacion, annot=True)
plt.show()







