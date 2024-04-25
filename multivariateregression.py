import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\ACER\OneDrive\Desktop\hiring.csv")
regression = linear_model.LinearRegression()
regression.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df[['salary($)']])
print(regression.predict([[2,9,6]]))
print(regression.predict([[12,10,10]]))
print(regression.coef_)
print(regression.intercept_)
X = df[['salary($)']]
Y = regression.predict(df[['experience','test_score(out of 10)','interview_score(out of 10)']])
print('R2 value is:{} '.format(r2_score(X,Y)*100))
#figure = plt.figure()
#ax = figure.add_subplot(1,2,1 , projection="3d")
#x = np.array(df['experience'])
#z = np.array(df['interview_score(out of 10)'])
#ax.plot3D(x*2812.95487627, y*1845.70596798, z*2205.24017467 + 17737.26346434, 'green')
#ax = figure.add_subplot(1,2,2 , projection="3d")
#plt.scatter(df['experience'],df['test_score(out of 10)'],df['interview_score(out of 10)'] )
#ax.set_title('regression model', fontsize=30)
plt.scatter(regression.predict(df[['experience','test_score(out of 10)','interview_score(out of 10)']]),df[['salary($)']])
plt.plot(regression.predict(df[['experience','test_score(out of 10)','interview_score(out of 10)']]),df[['salary($)']], '--')
plt.show()
