# importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

# train simple linear reg model on train set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predict test set result
y_pred = regressor.predict(x_test)

# visualise train set result
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs. Experience (Training Set)')
plt.xlabel('Experience (In Years)')
plt.ylabel('Salary')
plt.show()

# visualise test set result
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs. Experience (Test Set)')
plt.xlabel('Experience (In Years)')
plt.ylabel('Salary')
plt.show()
