import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

df = pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\gangtoktest.csv")
X = np.array(df.iloc[:, 0:3])
y = np.array(df.iloc[:, 3:4])
A = np.ones([X.shape[1]])
b = np.array([1.0])
G = -np.eye(X.shape[1])
h = np.zeros(X.shape[1])
X = matrix(X)
y = matrix(y)
Q = X.T*X
c = -X.T*y
P, q = matrix(Q), matrix(c)
G, h = matrix(G), matrix(h)
A = matrix(A)
A, b = A.T, matrix(b)

sol = solvers.qp(P, q, G, h, A, b)
coefficients = sol['x']
objective = sol['primal objective']
print(coefficients)

#print(objective)




