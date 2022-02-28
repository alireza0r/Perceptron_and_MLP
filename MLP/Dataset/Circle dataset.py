# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:57:23 2022

@author: alire
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_circle_data(r=9, n=100):
  x = np.linspace(-(r-0.1),(r-0.1), num=n//2)

  y_p = np.sqrt(r**2-x**2)
  y_n = -np.sqrt(r**2-x**2)

  x = np.reshape(x, (1,-1))
  y_p = np.reshape(y_p, (1,-1))
  y_n = np.reshape(y_n, (1,-1))

  Y_n = np.concatenate((x,y_n))
  Y_p = np.concatenate((x,y_p))

  XY = np.concatenate((Y_n,Y_p), axis=1)

  return XY

XY_circle1 = make_circle_data(9,50)
XY_circle2 = make_circle_data(1,50)

plt.plot(XY_circle1[0,:], XY_circle1[1,:], 'o')
plt.xlabel('f1')
plt.plot(XY_circle2[0,:], XY_circle2[1,:], 'o')
plt.ylabel('f2')

Y_t = np.concatenate((np.ones((1, 50)), np.zeros((1, 50))), axis=1)
Y_f = np.concatenate((np.zeros((1, 50)), np.ones((1, 50))), axis=1)

X = np.concatenate((XY_circle1, XY_circle2), axis=1)
Y = np.concatenate((Y_t, Y_f), axis=0)

input = X
target = Y

print(input.shape)
print(target.shape)

index = ['f1', 'f2']

pd.DataFrame(X, index = index).to_csv('x_dataset.csv')
pd.DataFrame(Y, index = index).to_csv('y_dataset.csv')

