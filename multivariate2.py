import pandas as pd
from sklearn import linear_model
from sklearn import constrained_linear_regression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\ACER\OneDrive\Desktop\pvdata.csv")
regression = linear_model.LinearRegression()
regression.fit(df[['V','V2']],df[['P']])
print(regression.coef_)
print(regression.intercept_)
X = df[['P']]
Y = regression.predict(df[['V','V2']])
print('R^2 value is:{} '.format(r2_score(X,Y)*100))
#figure = plt.figure()
#ax = figure.add_subplot(1,2,1 , projection="3d")
#x = np.array(df['V'])
#y = np.array(df['V2'])
#ax.plot3D(x*80.70771426, y*(-0.12899699) - 10930.36615902, 'green')
#ax = figure.add_subplot(1,2,2 , projection="3d")
#plt.scatter( , 'blue')
#ax.set_title('regression model', fontsize=20)
plt.plot(df[['V']],regression.predict(df[['V','V2']]), color='red')
plt.scatter(df[['V']],df[['P']])
plt.show()