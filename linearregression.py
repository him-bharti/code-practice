import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\ACER\OneDrive\Desktop\data.csv")
regression = linear_model.LinearRegression()
regression.fit(df[['year']],df[['per capita income (US$)']])
print(regression.predict(np.array([[2020]])))
print(regression.coef_)
print(regression.intercept_)
plt.scatter(df['year'],df['per capita income (US$)'])
plt.plot(df['year'],regression.predict(df[['year']]), color = 'blue')
plt.xlabel('Year')
plt.ylabel('per capita income (US$)')
plt.title('regression model prediction', fontsize=20)
plt.show()
p = list(regression.predict(np.array([[2020],[2021],[2022],[2023],[2024],[2025]])))
temp = []
for i in p:
    temp.append(float(i))
#print(temp)

year = ('2020','2021','2022','2023','2024','2025')
df1 = pd.DataFrame()
df1['Year'] = year
df1['per capita income (US$)'] = temp
print(df1)
df1.to_csv("C:\\Users\ACER\OneDrive\Desktop\data1.csv")