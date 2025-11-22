
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Checking for Titanic dataset...")


csv_file = None
for file in os.listdir():
    if file.lower().endswith(".csv"):
        csv_file = file
        break

if not csv_file:
    print("No CSV file found in the folder! Please add Titanic-Dataset.csv.")
    exit()

print(f"Found dataset: {csv_file}")


df = pd.read_csv(csv_file)
print("\nData Loaded Successfully!\n")
print(df.head())


print("\nMissing Values:\n", df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True, errors='ignore')


le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])


X = df.drop('Survived', axis=1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(8, 5))
sns.barplot(x=model.feature_importances_, y=X.columns, palette="Blues_d")
plt.title("Feature Importance in Survival Prediction")
plt.show()


plt.figure(figsize=(10, 4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival Count by Gender")
plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='viridis')
plt.title("Survival Count by Passenger Class")
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Passengers")
plt.show()

print("\nTitanic Survival Prediction completed successfully!")
