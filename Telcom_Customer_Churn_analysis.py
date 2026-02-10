from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv("data/Telco-Customer-Churn.csv")
df.dtypes
df
df=df.drop(columns=['customerID'])
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df['TotalCharges'].isnull().sum()
df = df[df.TotalCharges.notnull()]
#Data Visualization
##tenure
tenure_churn_no = df[df.Churn=='No'].tenure
tenure_churn_yes = df[df.Churn=='Yes'].tenure

plt.figure(figsize=(10, 6))
plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes, tenure_churn_no], bins=20, rwidth=0.95, color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()
##monthlycharges
tenure_churn_no = df[df.Churn=='No'].MonthlyCharges
tenure_churn_yes = df[df.Churn=='Yes'].MonthlyCharges

plt.figure(figsize=(10, 6))
plt.xlabel("monthly charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes, tenure_churn_no], bins=20, rwidth=0.95, color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()

##monthlycharges
tenure_churn_no = df[df.Churn=='No'].TotalCharges
tenure_churn_yes = df[df.Churn=='Yes'].TotalCharges
plt.figure(figsize=(10, 6))
plt.xlabel("total charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes, tenure_churn_no], bins=20, rwidth=0.95, color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()
##gender
tenure_churn_no = df[df.Churn=='No'].gender
tenure_churn_yes = df[df.Churn=='Yes'].gender

plt.figure(figsize=(10, 6))
plt.xlabel("gender")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes, tenure_churn_no], bins=20, rwidth=0.95, color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='str':
             print(f'{column}: {df[column].unique()}') 

print_unique_col_values(df)


df.replace('No internet service', value='No', inplace=True) 
df.replace('No phone service', value='No', inplace=True)

print_unique_col_values(df)

df.replace({'Yes': 1,'No': 0},inplace=True) 
df.replace({'Female': 1,'Male': 0},inplace=True) 

# ✓ Convertir les colonnes en int/float
df = df.astype(int)  

for col in df.columns:
    print(f'{col}: {df[col].unique()}') 

df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])
df.columns

#scaling numeric columns
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
df[['tenure','MonthlyCharges','TotalCharges']] = minmaxscaler.fit_transform(df[['tenure','MonthlyCharges','TotalCharges']])

#spliting data to train and test
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df.shape
X_train.shape
#creating ann model
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Input(shape=(26,)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)
yp=model.predict(X_test)
yp[:5]
#evaluation report
from sklearn.metrics import confusion_matrix , classification_report
print(confusion_matrix(y_test, yp.round()))
print(classification_report(y_test, yp.round()))

#confusion matrix

import seaborn as sns

cm = confusion_matrix(y_test, yp.round())
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()