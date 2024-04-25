import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv('data_set.csv')
print(dataset[['time', 'V_bus', 'Q_load', 'P_load', 'Q_br']])

# %matplotlib inline
plt.scatter(dataset.V_bus, dataset.P_load, s=0.01)
plt.xlabel('voltage(kV) ')
plt.ylabel('load(MW)')
plt.title('Correlation between V and P(at ICTs) at 400kV Subhasgram S/S', fontsize=10)

# %matplotlib inline
plt.scatter(dataset.V_bus, dataset.Q_load, s=0.01)
plt.xlabel('voltage(kV) ')
plt.ylabel('load(MVAR)')
plt.title('Correlation between V and Q (at ICTs)  at 400kV Subhasgram S/S', fontsize=10)

# %matplotlib inline
plt.scatter(dataset.P_load, dataset.Q_load, s=0.01)
plt.xlabel('P_load (KW) ')
plt.ylabel('Q_load (MVAR)')
plt.title('Correlation between P and Q (at ICTs)  at 400kV Subhasgram S/S', fontsize=10)

forcast_data = pd.read_csv('forcast.csv')
print(forcast_data)

reg_model2 = linear_model.LinearRegression()
x = dataset[['P_load', 'Q_load', 'Q_br']]
y = dataset.V_bus
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
reg_model2.fit(X_train, y_train)

input_x = forcast_data[['p_load', 'q_load', 'q_br']]
output_v = forcast_data.V_bus
V_predict = reg_model2.predict(input_x)

# import matplotlib.pyplot as plt
# forcast_data['time']=pd.to_datetime(forcast_data['time'])
plt.plot(forcast_data.time, V_predict)
plt.plot(forcast_data.time, forcast_data.V_bus)
plt.ylim(350, 420)
plt.xlabel('Time(24Hrs)')
plt.ylabel('Voltage (kV) ')
plt.title('Predicted Voltage vs Actual voltage',fontsize=10)
plt.grid(axis='y')
r2_score(forcast_data.V_bus, V_predict)

forcast_data_br_0 = pd.read_csv('forcast_1.csv')
forcast_data_br_1 = pd.read_csv('forcast_2.csv')

input_x = forcast_data_br_0[['p_load', 'q_load', 'q_br']]
output_v = forcast_data_br_0.V_bus
V_predict_br_0 = reg_model2.predict(input_x)

input_x = forcast_data_br_1[['p_load', 'q_load', 'q_br']]
output_v = forcast_data_br_1.V_bus
V_predict_br_1 = reg_model2.predict(input_x)

plt.plot(forcast_data_br_0.time, V_predict_br_0, label='BR=off from 00:00 to 24:00 Hrs', )
plt.plot(forcast_data_br_0.time, V_predict_br_1, label='BR=on from 02:00 Hrs to 08:00 Hrs', linewidth=0.8)

plt.ylim(350, 420)
plt.legend()
plt.xlabel('Time (24 Hrs) ')
plt.ylabel('Voltage(kV)')
plt.title('Voltage comparison with/without BR operation ', fontsize=10)
plt.grid(axis='y')