## Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\ml-projects\Churn Modelling\Churn_Modelling.csv')

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

## Creating dummy variables bcz geography and gender are categorical features
geography = pd.get_dummies(X["Geography"], drop_first=True)
gender = pd.get_dummies(X["Gender"], drop_first=True)

## Concatinating in the data frame
X = pd.concat([X, geography, gender], axis = 1)

## Drop un-necessary Columns
X = X.drop(['Geography', 'Gender'], axis = 1)

## Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Starting of ANN ##

## Importing keras libraries and packages
import keras 
from keras.models import Sequential ## Responsible in creating the NN
from keras.layers import Dense ## for hidden layers
from keras.layers import Dropout ## for regularization. to avoid over-fitting.

## Initializing ANN
classifier = Sequential()

## Adding Input Layer and First hidden layer
classifier.add(Dense(units = 6, kernel_initializer='he_uniform', activation='relu', input_dim=11))

## Adding Second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='he_uniform', activation='relu'))

## Adding Output layer
classifier.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation='sigmoid'))

## Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, nb_epoch =100)

print(model_history.history.keys)

## Making prediction and evaluating the model

# Predicting the Test data result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Calculate the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)