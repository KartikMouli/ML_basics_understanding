# importing libraries
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing database
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# print x y
# print(x)
# print(y)

y = y.reshape(len(y), 1)

# print y
# print(y)

# Feature scaling
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# print x y
# print(x)
# print(y)

# training svr model
regressor = SVR(kernel='rbf')
regressor.fit(x, y)


# Predict new result
# print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# Visualize svr result
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(
    regressor.predict(x)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
