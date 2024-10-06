import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_excel('C:/Users/alnaseem/Desktop/Ai course/lesson 4 (machine learning)/3-polynomial-regression/polynomial-regression.xlsx')

x=data.iloc[:,:-1]
y=data.iloc[:,-1:]

plt.scatter(x, y)
plt.show()

poly=PolynomialFeatures(degree=4)

p_x=poly.fit_transform(x)

m=LinearRegression()
m.fit(p_x, y)

print(m.score(p_x, y))

plt.scatter(x, y)

plt.plot(x,m.predict(p_x))

plt.show()


         
