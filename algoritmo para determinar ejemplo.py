#Primero Correr esto
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

dataset=pd.read_csv("C:/HR_comma_sep.csv")

#fijate en la tab de variable explorer 

#Vamos a cambiar las categorias a numeros
le = LabelEncoder()
dataset['Departments'] = le.fit_transform(dataset['Departments'])
dataset['salary'] = le.fit_transform(dataset['salary'])

#aqui checate el dataset

#Aqui preprocesas tus datos, y empiezas a hacer el modelo
y=dataset['left']
#valores que influyen cuando la persona se va 
features = ['satisfaction_level', 'last_evaluation', 'number_project',
'average_montly_hours', 'time_spend_company', 'Work_accident',
'promotion_last_5years', 'Departments', 'salary']
X=dataset[features]

#Despues de esto checate x, y , features


#vamos a escalar los datos, ponerlos en el mismo tipo a decimales
s = StandardScaler()
X = s.fit_transform(X)

#vamos a dividir y entrenar el dataset
X_train,X_test,y_train,y_test = train_test_split(X,y)

#vamos a dejar que el modelo predija resultados , es decir lo entrenamos
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X)
y_prob = log.predict_proba(X)

# Vamos a cargar las ultimas columnas para predicion en el dataset final
dataset['predictions'] = y_pred
dataset['probability of leaving'] = y_prob[:,1]