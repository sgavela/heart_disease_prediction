import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score

#ubicacion del dataset
file_path=  "D:/Sergi2/Tfg mates/data/heart.csv"

#leer los datos del archivo .csv
data = pd.read_csv(file_path)

#renombrar para mejor entendimiento
data.rename(columns={'age':'edad', 
                     'sex':'sexo', 
                     'cp':'tipo de dolor pectoral',
                     'trestbps':'tension en reposo', 
                     'chol': 'colesterol',
                     'fbs':'glucemia en ayunas', 
                     'restecg':'electrocardiograma',
                     'thalach':'ppm maximas',
                     'exang':'angina inducida',
                     'oldpeak':'depresion ST',
                     'slope':'pendiente',
                     'ca':'nº vasos mayores',
                     'thal': 'thal'
                     },                    
            inplace=True)

X = data.loc[:, data.columns != 'target']
y = data.loc[:, 'target']


model = GaussianNB()

#Cálculo de las métricas usando crossvalidation cv=5
scores_precision = cross_val_score(model, X, y, cv=5, scoring='precision')
print("Scores precision:", scores_precision)
scores_recall = cross_val_score(model, X, y, cv=5, scoring='recall')
print("Scores recall:", scores_recall)
scores_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
print("Scores f1:", scores_f1)
scores_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print("Scores auc:", scores_auc)

print('\n')

print("Mean precision:", round(scores_precision.mean(),4))
print("Mean recall:", round(scores_recall.mean(),4))
print("Mean f1:", round(scores_f1.mean(),4))
print("Mean auc:", round(scores_auc.mean(),4))

print('\n')

#Cálculo de las métricas separando train y test aleatoriamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

recall = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print("Accuracy:",round(recall,4))
print("Precision:",round(precision,4))

f1_score = (2*precision*recall)/(precision+recall)
print("F1 score: ", round(f1_score,4))

y_test = y_test.to_numpy()

tpr = []
fpr = []
thresholds = [0, 0.5, 1]
for i in thresholds:
    tp = 0
    fp = 0   
    for j in range(len(y_prob)):
        if y_prob[j][1] > i and y_test[j] == 1:
            tp += 1
        elif y_prob[j][1] > i and y_test[j] == 0:
            fp += 1
    
    if i == 0:
        tpr.append(0)
        fpr.append(0)
    elif tp+fp != 0:    
        tpr.append(tp/(tp+fp))
        fpr.append(fp/(tp+fp))
    else:
        tpr.append(1)
        fpr.append(1)
        
auc = 0
j = 0
for i in range(1,len(thresholds)):
    auc += ((fpr[i]-fpr[i-1]) * (tpr[i] - tpr[i-1]))/2 
   
    auc += (fpr[i]-fpr[i-1]) * (fpr[i]-fpr[i-1])
print("auc:", auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = '+str(round(auc,4)) + ')')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
         label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive-Bayes ROC curve for arbitrary partition')
plt.legend(loc="lower right")
plt.show()

'''
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test[i], y_pred[i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
'''
y_prob = y_prob[:,1]
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, y_prob)
auc2 = metrics.roc_auc_score(y_test, y_pred)
print("auc2:", auc2)
plt.figure()
lw = 2
plt.plot(fpr2, tpr2, color='darkorange',
         lw=lw, label='ROC curve (auc = '+str(round(auc2,4)) + ')')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
         label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive-Bayes ROC curve for arbitrary partition')
plt.legend(loc="lower right")
plt.show()