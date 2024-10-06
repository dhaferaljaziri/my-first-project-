import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_excel('C:/Users/alnaseem/Desktop/Ai course/lesson 4 (machine learning)/3-polynomial-regression/polynomial-regression.xlsx')

x=data.iloc[:,:-1]
y=data.iloc[:,-1:]

plt.scatter(x, y)
plt.show()

m=LinearRegression()
m.fit(x,y)

print(m.score(x, y))

plt.scatter(x, y)
plt.plot(x,m.predict(x))
plt.show()

#في هذا النوع من البيانات لا ينفع linear regression 
#لان best of line بعيد جدا عن القيم البيانات المحدده




