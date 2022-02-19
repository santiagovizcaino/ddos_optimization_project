# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#data = pd.read_csv(r"C:\Users\santy\Desktop\dataset_articulo.csv", encoding = 'utf-8') # acc=1
data = pd.read_csv(r"C:\Users\santy\Desktop\dataset_003.csv", encoding = 'utf-8') # acc=093
#CATEGORIZACION DE VARIABLES
#data['ack']=np.where(data['ack']>0,1,0)
data['type'] = pd.factorize(data.type)[0]
#data['source'] = pd.factorize(data.source)[0]
#data['protocol'] = pd.factorize(data.protocol)[0]
data.head()


#DIVISION DEL DATASET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_data=train_test_split(data,data.type,test_size=0.3,random_state=10)

#LOGIST REGRESION

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

model.score(x_test,y_test)

model.predict_proba(x_test)

#resultados
def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

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
sn.heatmap(cm, annot=labels, fmt='', cmap="Oranges")



# labels = ['True Neg','False Pos','False Neg','True Pos']
# labels = np.asarray(labels).reshape(2,2)
# plt.figure(figsize=(10,7))
# sn.heatmap(cm/np.sum(cm),annot=labels,fmt='', cmap='Blues')
# plt.xlabel("Predicciones")
# plt.ylabel("Verdades")


