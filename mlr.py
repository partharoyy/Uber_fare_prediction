# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# reading dataset
data = pd.read_csv('taxi.csv')

# print(data.head()) -- To check the dataset

# defining dependent and independent variables
data_x = data.iloc[:, 0:-1].values
data_y = data.iloc[:, -1].values

# print(data_y) -- To check the values

# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, random_state=0)

# training the data
reg = LinearRegression()
reg.fit(X_train, y_train)

# checking score
print('Train score: ', reg.score(X_train, y_train))
print('Train score: ', reg.score(X_test, y_test))

# creating pickle file
pickle.dump(reg, open('taxi.pkl', 'wb'))

# creating model
model = pickle.load(open('taxi.pkl', 'rb'))

# predicting result
print(model.predict([[80, 1770000, 6000, 85]]))
