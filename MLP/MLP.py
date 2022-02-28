# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 00:20:15 2022

@author: AlirezaRahmati
"""

import numpy as np
import matplotlib.pyplot as plt

#from tqdm import tqdm

'''
z = Wx + b
y_pred = act(z)
l = loss(y_pred)
s = dl*dy , sigma
'''
class Perceptron():
  def __init__(self, act, loss, feature_shape):
    self.param = {'w': np.random.randn(1, feature_shape) * 2,
                  'b': np.array([[0]]),#np.random.randn(1, 1),
                  'z':0,
                  'y_pred':0,
                  'dw':0,
                  'db':0,
                  's':0,
                  'da':0,
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

  def get_dloss(self, y):
    dl = self.dloss_used(y, self.param['y_pred'])
    return dl

  def backward(self, x, d_loss_value):
    #dl = self.dloss_used(y, self.param['y_pred'])
    da = self.dact_used(self.param['z'])
    self.param['da'] = da
    self.param['s'] = d_loss_value*da
    self.param['dw'] = d_loss_value*da*x
    self.param['db'] = d_loss_value*da*1
    return (self.param['dw'], self.param['db'], self.param['s'])

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
      d_loss_value = self.get_dloss(y)
      self.backward(x, d_loss_value)
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
    plt.show()


'''
perceptron: perceptron model
activation_list_for_each_layer: activation list of each layer
feature_shape: input shape
neurons_in_layer: number of neuron for each layer
'''
class MLP():
  def __init__(self, perceptron, activation_list_for_each_layer, feature_shape, neurons_in_layer):
    self.feature_shape = feature_shape
    self.neurons_in_layer = neurons_in_layer
    self.layer = len(neurons_in_layer)
    self.perceptron = perceptron
    self.activation_list_for_each_layer = activation_list_for_each_layer

    self.model_dict = self.CreateModel()

    self.decision = 0.5 if activation_list_for_each_layer[-1] == 'sigmoid' else 0

  # Create MLP model
  def CreateModel(self):
    model_dict = {}
    for l in range(self.layer):
      for n in range(self.neurons_in_layer[l]):
        if l == 0:
          #model_dict['layer'+str(l+1)+'-'+str(n+1)] = Perceptron('tanh', 'mse', self.feature_shape)
          model_dict['layer'+str(l+1)+'-'+str(n+1)] = Perceptron(self.activation_list_for_each_layer[l], 'mse', self.feature_shape)
        else:
          model_dict['layer'+str(l+1)+'-'+str(n+1)] = Perceptron(self.activation_list_for_each_layer[l], 'mse', self.neurons_in_layer[l-1])
    return model_dict

  # Forward data
  def forward(self, input):
    x_for_layer = input
    self.out_put_of_each_layer_dict = {'layer0':x_for_layer}
    for l in range(self.layer):
      y_in_layer = np.array([])
      for n in range(self.neurons_in_layer[l]):
        if y_in_layer.shape[0] == 0:
          y_in_layer = self.model_dict['layer'+str(l+1)+'-'+str(n+1)].forward(x_for_layer)
        else:
          y_in_layer = np.concatenate((y_in_layer, self.model_dict['layer'+str(l+1)+'-'+str(n+1)].forward(x_for_layer)))

      x_for_layer = y_in_layer

      self.out_put_of_each_layer_dict['layer'+str(l+1)] = y_in_layer
    return y_in_layer

  # Backward
  def backward(self, target):
    out_put_loss = []
    backward_sigma = {}
    for l in range(self.layer-1, -1, -1):
      if l>0:
        backward_sigma['layer'+str(l+1)] = np.zeros(shape=(self.neurons_in_layer[l-1], target.shape[-1]))
      else:
        backward_sigma['layer'+str(l+1)] = np.zeros(shape=(self.feature_shape, target.shape[-1]))

      x_for_layer = self.out_put_of_each_layer_dict['layer'+str(l)]
      if l == self.layer-1:
        for n in range(self.neurons_in_layer[l]):
          #print(f'backward:{l+1}-{n+1}')

          pc = self.model_dict['layer'+str(l+1)+'-'+str(n+1)]

          d_loss_value = pc.get_dloss(target[n, :])
          sigma = pc.backward(x_for_layer, d_loss_value)[2]
          w_s = np.dot(pc.param['w'].T, sigma)

          backward_sigma['layer'+str(l+1)] += w_s

          out_put_loss = d_loss_value.sum(-1)[0]
      else:
        sigma = backward_sigma['layer'+str(l+1+1)]

        for n in range(self.neurons_in_layer[l]):
          #print(f'backward:{l+1}-{n+1}')
          pc = self.model_dict['layer'+str(l+1)+'-'+str(n+1)]

          d_loss_value = sigma[n,:]
          sigma_n = pc.backward(x_for_layer, d_loss_value)[2]

          w_s = np.dot(pc.param['w'].T, sigma_n)

          backward_sigma['layer'+str(l+1)] += w_s
    return out_put_loss

  # Updating
  def update(self, learning_rate):
    for l in range(self.layer):
      for n in range(self.neurons_in_layer[l]):
        pc = self.model_dict['layer'+str(l+1)+'-'+str(n+1)]
        pc.update(learning_rate) 

        #print('--------------------------')
        #print(f'Updating:{l+1}-{n+1}')
        #print(pc)

  # Run
  def __call__(self, x, y, learning_rate, epochs, batch_size=1, **arg):
    out_loss_list = []
    valid_list = []
    #for iter in tqdm(range(epochs)):
    for iter in range(epochs):
      (x_batch, y_batch) = self.create_batch(x, y, batch_size)

      for i in range(len(x_batch)):
        x_new = x_batch[i]
        y_new = y_batch[i]

        self.forward(x_new)
        out_loss = self.backward(y_new)
        self.update(learning_rate)
        out_loss_list.append(out_loss)
      
      if 'x_valid' in arg.keys():
        valid_list.append(self.valid(arg['x_valid'], arg['y_valid']))

    return valid_list, out_loss_list

  def create_batch(self, x, y, batch_size):
    data_len = x.shape[1]
    index_list = np.arange(data_len)
    np.random.shuffle(index_list)

    x_batchs_list = []
    y_batchs_list = []

    x_batch = []
    y_batch = []
    for i, indx in enumerate(index_list):
      j = i+1
      x_batch.append(x[:, indx])
      y_batch.append(y[:, indx])

      if j%batch_size == 0:
        x_batchs_list.append(np.array(x_batch).T)
        y_batchs_list.append(np.array(y_batch).T)

        x_batch = []
        y_batch = []
      elif j == data_len:
        x_batchs_list.append(np.array(x_batch).T)
        y_batchs_list.append(np.array(y_batch).T)

        x_batch = []
        y_batch = []

    return (x_batchs_list, y_batchs_list)

  def print_param(self):
    for l in range(self.layer):
      for n in range(self.neurons_in_layer[l]):
        text = 'layer'+str(l+1)+'-'+str(n+1)
        print(text)
        print('w:')
        print(self.model_dict['layer'+str(l+1)+'-'+str(n+1)].param['w'])
        print('b:')
        print(self.model_dict['layer'+str(l+1)+'-'+str(n+1)].param['b'])

  def predict(self, x):
    y_pred = self.forward(x)
    return y_pred > self.decision

  def mse_calculator(self, y, y_pred):
    return ((y - y_pred)**2).sum(-1) / y.shape[-1]

  def valid(self, x, y):
    y_pred = self.predict(x)
    return self.mse_calculator(y, y_pred)
