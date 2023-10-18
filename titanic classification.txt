import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset (assuming you have a CSV file)
titanic_data = pd.read_csv('titanic_dataset.csv')

# Data preprocessing
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Sex'] = LabelEncoder().fit_transform(titanic_data['Sex'])
titanic_data = pd.get_dummies(titanic_data, columns=['Pclass'], prefix=['Pclass'])

# Define the features and target variable
X = titanic_data[['Pclass_1', 'Pclass_2', 'Age', 'Sex']]
y = titanic_data['Survived']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model (logistic regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
