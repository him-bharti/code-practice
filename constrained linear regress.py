from constrained_linear_regression import ConstrainedLinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
df = pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\pvdata3.csv")
X = np.array(df.iloc[:, 0:2])
y = np.array(df.iloc[:, 2:3])
y = y.ravel()
model = ConstrainedLinearRegression(nonnegative=True)

#max_coef = np.repeat(1, X.shape[1])
#max_coef = np.array([1])#, X.shape[1])
model.fit(X, y)#, max_coef= max_coef)
print(model.intercept_,model.coef_)
print(r2_score(y,model.predict(X)))

