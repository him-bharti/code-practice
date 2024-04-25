import cvxpy as cp
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

df = pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\pvdata3.csv")
X = np.array(df.iloc[:, 0:2])
y = np.array(df.iloc[:, 2:3])
beta = cp.Variable(X.shape[1])
A = np.ones([X.shape[1]])
b = np.array([1.0])
G = -np.eye(X.shape[1])
h = np.zeros(X.shape[1])
#constraints = [A @ beta == b]
X = matrix(X)
y = matrix(y)
Q = X.T*X
c = -X.T*y
P = matrix(Q)
q = matrix(c)
G = matrix(G)
h = matrix(h)
A = matrix(A)
A = A.T
b = matrix(b)

sol = solvers.qp(P, q, G, h, A, b)
coefficients = sol['x']
objective = sol['primal objective']
print(coefficients)
print(objective)
print(Q)

