import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

#visualizar las primeras cinco filas
dataset_head=data.head()
print(dataset_head)

#visualizar edad
x_values = data['edad'].unique()
y_values = data['edad'].value_counts().tolist()
plt.bar(x_values, y_values)
plt.title('Edad')
plt.show()

#visualizar sexo
y_values = data['sexo'].value_counts().tolist()
plt.pie(y_values, labels=["Varón", "Mujer"], autopct="%0.1f %%")
plt.title('Sexo')
plt.show()

#visualizar target
y_values = data['target'].value_counts().tolist()
plt.pie(y_values, labels=[
    "Presencia de enfermedad", 
    "Ausencia de enfermedad"], 
    autopct="%0.1f %%")
plt.title('Target')
plt.show()


#graficar las dependencias entre características
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(method='pearson'),annot=True,square=True)
plt.show()

#visualizar el numero de valores nulos en cada columna
for column in data.columns.tolist():
    print ("Valores nulos en <{0}>: {1}".
           format(column, data[column].isnull().sum()))

#visualizar algunos indicadores estadísticos 
#(media, varianza, min, max) de cada columna
data_description=data.describe()
print(data_description)

    