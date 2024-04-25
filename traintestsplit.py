import numpy as np
from scipy.optimize import fmin_slsqp
import pandas as pd

def linear_regression_loss(params, X, y):
    # Unpack the coefficients
    a, b, c = params

    # Calculate the predicted values
    y_pred = a * X + b * X**2 + c

    # Calculate the mean squared error
    mse = np.mean((y - y_pred)**2)

    return mse

def constraint_func(params):
    # Define your constraints here
    # For example, let's set bounds for 'a' and 'b'
    a, b, c = params
    #lower_bound_a = 0
    #upper_bound_a = 2
    #lower_bound_b = 0
    #upper_bound_b = 10
    #lower_bound_c = 0


    # Return the constraints as a 1-D array
    return np.array([a + b + c == 1])#- lower_bound_a,b - lower_bound_b, c - lower_bound_c])

if __name__ == "__main__":
    # Example data (replace this with your data)
    df = pd.read_csv("C:\\Users\ACER\OneDrive\Desktop\pvdata1.csv")
    X = np.array(df.iloc[:, 0:1])
    X = X.flatten()
    y = np.array(df.iloc[:, 1:2])
    y = y.flatten()

    # Initial guess for the parameters
    initial_params = np.array([1,1,1])

    # Perform constrained optimization
    optimized_params = fmin_slsqp(linear_regression_loss, initial_params, eqcons=[constraint_func], args=(X, y))

    # Unpack the optimized coefficients
    a_opt, b_opt , c_opt = optimized_params

    print(f"Optimized 'a' coefficient: {a_opt}")
    print(f"Optimized 'b' coefficient: {b_opt}")
    print(f"Optimized 'c' coefficient: {c_opt}")