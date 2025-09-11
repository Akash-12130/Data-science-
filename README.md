# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: X = input feature, y = target variable
X = np.array([[1], [2], [3], [4], [5]])  # 2D array for sklearn
y = np.array([2, 4, 5, 4, 5])           # Target values

# Create the model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Print the coefficients
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
