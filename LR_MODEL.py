import numpy as np
from sklearn.linear_model import LinearRegression

data = np.loadtxt('physical.txt',delimiter='\t',skiprows=1)



y = data[ : , [0]] # or data_from_file[ : , 0:1]
x = data[:, 1:] # or data_from_file[ : , 1:2]
# x
print(x)

lin = LinearRegression()

model = lin.fit(x,y)

coef = lin.coef_

print("The Coeff Weight is", coef)

intercept = lin.intercept_

print("The Intercept Weights are", intercept)

y_pred = model.predict(x)
print(y_pred)
print(y)
