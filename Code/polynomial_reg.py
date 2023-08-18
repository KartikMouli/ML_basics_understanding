# importing libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing database
database = pd.read_csv('Position_Salaries.csv')
x = database.iloc[:, 1:-1].values
y = database.iloc[:, -1].values

# train linear regg model
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# train polynomial regg model
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# visualise linear regg result
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salry')
plt.show()


# visualise poly regg result
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salry')
plt.show()

# predict new result with linear regg
print(lin_reg.predict([[6.5]]))

# predict new result with polynomial regg
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
