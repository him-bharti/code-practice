import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

# data
df = pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\pvdata3.csv")
X = np.array(df.iloc[:, 0:2])   # Features
y = np.array(df.iloc[:, 2:3])   # Dependent variable

# Add a column of ones for the intercept term
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

# Define equality constraint matrix A and vector b
A_eq = matrix([[1.0, -1.0, 0.0]])
A_eq = A_eq.T
A_neg = matrix(-np.identity(X_with_intercept.shape[1]))  # Non-negative constraint for all coefficients
A = matrix(np.vstack((A_eq, A_neg)))
b = matrix([1.0])


# Construct matrices for the quadratic programming problem
X_matrix = matrix(X)
y_matrix = matrix(y)

# Construct the matrices for the linear regression problem
H = X_matrix.T * X_matrix  # Quadratic term
f = -X_matrix.T * y_matrix  # Linear term

# Solve the quadratic programming problem
sol = solvers.qp(H, f, A=A ,b=b)

# Extract the solution (coefficients)
c = np.array(sol['x'])

print("Estimated coefficients:", c)


"""print(X_with_intercept[0:2,:])
print(matrix(-np.identity(X_with_intercept.shape[1])))
A_eq = matrix(np.array([1.0, 1.0, 1.0]))
A_eq = A_eq.T
A_neg = matrix(-np.identity(X_with_intercept.shape[1]))  # Non-negative constraint for all coefficients
A = matrix(np.vstack((A_eq, A_neg)))
b = matrix([1.0])
print(A_eq)
print(A_neg)
print(A)"""



