# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:00:27 2022

@author: AlirezaRahmati
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#x0 = np.expand_dims(np.linspace(-0.2,0.2), axis=0)
x10 = np.linspace(-0.2,0.2)
np.random.shuffle(x10)
x10 = np.expand_dims(x10, axis=0)

x11 = np.linspace(0.8,1.2)
np.random.shuffle(x11)
x11 = np.expand_dims(x11, axis=0)

x20 = np.linspace(-0.2,0.2)
np.random.shuffle(x20)
x20 = np.expand_dims(x20, axis=0)

x21 = np.linspace(0.8,1.2)
np.random.shuffle(x21)
x21 = np.expand_dims(x21, axis=0)

X1 = np.concatenate((x10, x11, x10, x11), axis=1)
X2 = np.concatenate((x20, x20, x21, x21), axis=1)

Y = np.concatenate((np.zeros((1,50)), np.ones((1,50)), np.ones((1,50)), np.zeros((1,50))), axis=1)

X = np.concatenate((X1, X2), axis=0)

print(X.shape)
print(Y.shape)

plt.plot(X[0,:], X[1,:], 'o')
plt.xlabel('f1')
plt.ylabel('f2')
plt.show()

print(X.shape)

index = ['f1', 'f2']
pd.DataFrame(X, index = index).to_csv('x_xor_dataset.csv')
pd.DataFrame(Y).to_csv('y_xor_dataset.csv')

print('file_saved')