import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
        'var_smoothing': np.logspace(0,-9, num=100)
    }

gs = GridSearchCV(
        GaussianNB(),
        grid_params,
        verbose = 0,
        cv = 4,
        scoring = 'f1',
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


