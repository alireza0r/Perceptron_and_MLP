# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 00:27:03 2022

@author: AlirezaRahmati
"""
import numpy as np

def shuffling(x, y):
  data_len =x.shape[-1]
  index_list = np.arange(0, data_len)
  np.random.shuffle(index_list)

  x_new = []
  y_new = []
  for i in index_list:
    x_new.append(x[:,i])
    y_new.append(y[:,i])

  return np.array(x_new).T, np.array(y_new).T

def mse_calculator(y, y_pred):
  return (((y - y_pred)**2).sum(-1) / y.shape[-1]).sum(0)

def error_calculator(y, y_pred, decision):
  y_pred = (y_pred > decision)*1
  return (np.abs((y - y_pred)).sum(-1)).sum(0)