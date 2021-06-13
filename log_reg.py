import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


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

grid_params = {
        'penalty': ['l2'],
        'C': [0,6158],
        'solver': ['liblinear']
    }

gs = GridSearchCV(
        LogisticRegression(),
        grid_params,
        verbose = 0,
        cv = 4,
        scoring = 'neg_mean_squared_error',
        n_jobs = -1
    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gs_results = gs.fit(X_train, y_train)

print("Best params")
for key in gs_results.best_params_:
    print (key, ":", gs_results.best_params_[key])
print('\n')  
print("Best score")
print(gs_results.best_score_)
print('\n')
print("Best estimator")  
print(gs_results.best_estimator_)
print('\n')
results = gs_results.cv_results_

y_prob = gs.predict_proba(X_test)
y_prob = y_prob[:,1]
y_test = y_test.to_numpy()

THRES = 0.5
y_pred = []
for prediction in y_prob:
    if prediction <= THRES:
        y_pred.append(0)
    else:
        y_pred.append(1)

recall = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print("Accuracy:",round(recall,4))
print("Precision:",round(precision,4))
f1_score = (2*precision*recall)/(precision+recall)
print("F1 score: ", round(f1_score,4))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
auc = metrics.roc_auc_score(y_test, y_prob)
print("auc:", round(auc,4))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (auc = '+str(round(auc,4)) + ')')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
         label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Linear Regression ROC curve')
plt.legend(loc="lower right")
plt.show()


