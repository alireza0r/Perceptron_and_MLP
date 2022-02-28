# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:38:33 2022

@author: AlirezaRahmati
"""

import numpy as np
import matplotlib.pyplot as plt

#from tqdm import tqdm

'''
z = Wx + b
y_pred = act(z)
l = loss(y_pred)
'''
class Perceptron():
  def __init__(self, act, loss, feature_shape):
    self.param = {'w': np.random.randn(1, feature_shape) * 0.01,
                  'b': np.random.randn(1, 1) * 0.01,
                  'z':0,
                  'y_pred':0,
                  'dw':0,
                  'db':0,
                  }

    self.dact_used = self.d_activation(act)
    self.act_used = self.activation(act)

    self.loss_used = self.loss(loss)
    self.dloss_used = self.d_loss(loss)

    self.loss_list = list()

  def regression(self, x):
    return np.dot(self.param['w'], x) + self.param['b']

  def activation(self, act):
    act_dict = {
                'sigmoid':  (lambda x: 1/(1 + np.exp(-x))),
                'tanh':     (lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))),
                'linear':   (lambda x: x),
                'relu':     (lambda x: x * (x > 0)),
               }
    return act_dict[act]

  def d_activation(self, act):
    act_dict = {
                'sigmoid':  lambda x: np.exp(-x)/(1 + np.exp(-x))**2,
                'tanh':     lambda x: 4/(np.exp(x) + np.exp(-x))**2,
                'linear':   lambda x: 1,
                'relu':     lambda x: 1 * (x > 0),
               }

    return act_dict[act]

  def loss(self, selsect_loss):
    l = {
        'loglikelihood': lambda y,y_pred: -y*np.log(y_pred) - (1-y)*np.log(1-y_pred),
        'mse': lambda y,y_pred: (y-y_pred)**2 / 2,
        }
    return l[selsect_loss]

  def d_loss(self, select_dloss):
    dloss = {
            #'loglikelihood': lambda y,y_pred: ((-1)/(y_pred))*(y==1) + ((-1)/(1-y_pred))*(y==0) + ((-1)/(-y_pred))*(y==-1),
            'loglikelihood': lambda y,y_pred: ((-y)/(y_pred)) + (-(1-y)/(1-y_pred)),
            'mse': lambda y,y_pred: y_pred-y
            }
    return dloss[select_dloss]

  def forward(self, x):
    self.param['z'] = self.regression(x)
    self.param['y_pred'] = self.act_used(self.param['z'])
    return self.param['y_pred']

  def backward(self, x, y):
    dl = self.dloss_used(y, self.param['y_pred'])
    da = self.dact_used(self.param['z'])
    self.param['dw'] = dl*da*x
    self.param['db'] = dl*da*1
    return (self.param['dw'], self.param['db'])

  def update(self, eta):
    dw_sum = self.param['dw'].sum(axis=-1)
    db_sum = self.param['db'].sum(axis=-1)

    self.param['w'] = self.param['w'] - eta*dw_sum
    self.param['b'] = self.param['b'] - eta*db_sum
    return (self.param['w'], self.param['b'])

  def mse_loss(self, y, y_pred):
    diff = (y-y_pred)**2
    return diff.sum(axis=-1) / y.shape[-1]

  def __call__(self, x, y, eta=0.1, iter=20, retrain=True, x_valid=None, y_valid=None):
    if retrain==True:
      self.param['w'] = np.random.randn(1, x.shape[0]) * 0.01
      self.param['b'] = np.random.randn(1, 1) * 0.01
      self.loss_list = []
    
    #for i in tqdm(range(iter)):
    for i in range(iter):
      self.forward(x)
      self.backward(x, y)
      self.update(eta)

      if x_valid.any() != None and y_valid.any() != None:
        self.loss_list.append(self.mse_loss(y, self.param['y_pred']).item())
    return self.loss_list

  def predict(self, x):
    return self.forward(x)

  def predict_class(self, x, s):
    pred = self.predict(x)
    classification = lambda x: (x>s)
    return classification(pred)

  def weight(self):
    return (self.param['w'], self.param['b'])

  def plot_line(self, x1):
    x2 = -((self.param['w'][0,0] / self.param['w'][0,1]) * x1 + (self.param['b'] / self.param['w'][0,1]))
    plt.plot(list(x1), list(x2[0,:]))
    plt.title('Separator line')
    plt.xlabel('X1')
    plt.ylabel('X2')
    #plt.show()


input = np.array([[0, 0],[0, 1],[1, 0],[1.1, 0.95], [1, 0.1], [1.1, 1], [1.2, 0.9], [0, 0]]).T # AND gate
target = np.array([[-1, -1, -1, 1, -1, 1, 1, -1]])

forward = Perceptron('tanh', 'mse', 2)
loss = forward(input, target, eta=0.1, iter=50, x_valid=input, y_valid=target)
print()

plt.subplot(2,1,1)
plt.plot(loss)
plt.xlabel('iter')
plt.ylabel('MSE loss (0-1)')
print(len(loss))

plt.subplot(2,1,2)
forward.plot_line(np.array([-0.2, 1.2]))
plt.plot(input[0,:], input[1,:], 'o')
plt.show()
