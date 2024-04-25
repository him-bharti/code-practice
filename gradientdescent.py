import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

def gradient_descent(x1,x2,x3,y):
    m_curr1 = m_curr2 = m_curr3 = b_curr = 0
    iterations = 100
    n = len(x1)
    learning_rate = 0.005

    for i in range(iterations):
        y_predicted = m_curr1 * x1 + m_curr2 * x2 + m_curr3 * x3 + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md1 = -(2/n) * sum(x1*(y-y_predicted))
        md2 = -(2/n) * sum(x2 * (y-y_predicted))
        md3 = -(2/n) * sum(x3 * (y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr1 = m_curr1 - learning_rate * md1
        m_curr2 = m_curr2 - learning_rate * md2
        m_curr3 = m_curr3 - learning_rate * md3
        b_curr = b_curr - learning_rate * bd
        print("m1 {}, m2 {}, m3 {}, b {}, cost {}, iteration {}".format(m_curr1,m_curr2,m_curr3,b_curr,cost, i))
        #plt.scatter(m_curr1, cost, color='blue')
        #plt.scatter(m_curr2, cost, color='green')
        #plt.xlabel('slope variation')
        #plt.ylabel('cost function/error function')


df = pd.read_csv("C:\\Users\ACER\OneDrive\Desktop\hiring.csv")
#x1 = np.array(df.iloc[:, 0:1])
#x1 = x1.flatten
x1 = np.array([1])
#x2 = np.array(df.iloc[:, 1:2])
#x2 = x2.flatten()
x2 = np.array([3])
#x3 = np.array(df.iloc[:, 2:3])
#x3 = x3.flatten()
x3 = np.array([5])
#y = np.array(df.iloc[:, 3:4])
#y = y.flatten()
y = np.array([12])
gradient_descent(x1, x2, x3, y)
#plt.legend(['m1','m2'])
#plt.show()