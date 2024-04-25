import numpy as np
from gekko import GEKKO
import pandas as pd
df = pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\PV.csv")
arr1 = np.array(df.iloc[:, 1:2])
arr2 = np.array(df.iloc[:, 2:3])
arr3 = np.array(df.iloc[:, 0:1])
arr1 = arr1.flatten()
arr2 = arr2.flatten()
arr3 = arr3.flatten()


# Method: Gekko for constrained regression
m = GEKKO(remote=False)
m.options.IMODE = 2
#c = m.Array(m.MV, 3)
c = [m.FV(value=0) for i in range(3)]
c[0].STATUS = 1
c[1].STATUS = 1
c[2].STATUS = 1

c[0].lower = 0
c[1].lower = 0
c[2].lower = 0

x1 = m.Param(value=arr1)
x2 = m.Param(value=arr2)
yd = m.CV(value=arr3)
#yd.FSTATUS = 1
yp = m.Var()
#yp1 = m.Var(lb=0, ub=1)

m.Equations([yp == c[0]*x1+c[1]*x2+c[2]])#,1 == c[0]+c[1]+c[2]])
m.Minimize((yd-yp)**2)

m.solve(disp=False)
coefficients = [c[0][0], c[1][0], c[2][0]]
print(coefficients)
#print(c[0])
#print(c[1])
#print(c[2])