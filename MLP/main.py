# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 00:28:30 2022

@author: AlirezaRahmati
"""
from MLP import *
from Utils import *

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='MultiLayer Perceptron Training')

parser.add_argument('--input-shape', type=int, help='input shape for model')
parser.add_argument('--output-shape', type=int, help='output shape for model')

parser.add_argument('--n-list', type=list, help='you should make a list of neuron numbers for each layer, for example use --N-list 2-3-1 for a MLP with 2 neurons in first and 3 neurons in second layer and 1 neuron for output layer')
parser.add_argument('--a-list', type=list, help='you should make a list of activation for each layer, for example use --A-list t-t-s for a MLP with tanh in first and second layer and sigmoid for output layer')

parser.add_argument('--l', type=float, help='Learning rate')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--batch', type=int, help='Batch size')

parser.add_argument('--x', type=list, help='input dataset address')
parser.add_argument('--y', type=list, help='output dataset address')

parser.add_argument('--x-valid', type=list, help='input dataset address for validation')
parser.add_argument('--y-valid', type=list, help='output dataset address for validation')

parser.add_argument('--x-test', type=list, help='input dataset address for test')
parser.add_argument('--y-test', type=list, help='output dataset address for test')

args = parser.parse_args()

input_shape = args.input_shape
output_shape = args.output_shape
learning_rate = args.l
epochs = args.epochs
batch_size = args.batch

x = args.x
y = args.y
if x != None:
    x = ''.join(x)
    y = ''.join(y)
    
    x = pd.read_csv(x).copy()
    y = pd.read_csv(y).copy()
    
    x = x.values[:,1:].astype(float)
    y = y.values[:,1:].astype(float)

x_valid = args.x_valid
y_valid = args.y_valid
if x_valid != None:
    x_valid = ''.join(x_valid)
    y_valid = ''.join(y_valid)
    
    x_valid = pd.read_csv(x_valid).copy()
    y_valid = pd.read_csv(y_valid).copy()
    
    x_valid = x_valid.values[:,1:].astype(float)
    y_valid = y_valid.values[:,1:].astype(float)

n_list = ''.join(args.n_list)
a_list = ''.join(args.a_list)

n_list =[int(n) for n in n_list.split('-')]
a_list = a_list.split('-')

if len(n_list) != len(a_list):
    raise ValueError('a_list (activation) and n_list (number of neuron for each layer) must to have the same size...!')

if __name__ == '__main__':
    new_a_list = []
    for a in a_list:
        if a == 's':
            new_a_list.append('sigmoid')
        elif a == 't':
            new_a_list.append('tanh')
        elif a == 'l':
            new_a_list.append('linear')
        elif a == 'r':
            new_a_list.append('relu')
        else:
            raise ValueError('please use activation of [s: sigmoid, t: tanh, r:relu, l:linear]...!')
    

    mlp_model = MLP(Perceptron, 
                    activation_list_for_each_layer = new_a_list, 
                    feature_shape = input_shape,
                    neurons_in_layer = n_list)

    result = mlp_model(x, y, learning_rate, epochs, batch_size, x_valid=x_valid, y_valid=y_valid)
    
    print('------------------------------------------------------------------------')
    out = mlp_model.forward(x)
    print(f'MSE error for train dataset:{mse_calculator(y, out)}%')
    print(f'number of error for train dataset:{error_calculator(y, out, 0.5)}')
    
    print('------------------------------------------------------------------------')
    out_valid = mlp_model.forward(x_valid)
    print(f'MSE error for valid dataset:{mse_calculator(y_valid, out_valid)}%')
    print(f'number of error for valid dataset:{error_calculator(y_valid, out_valid, 0.5)}')
    
    x_test = args.x_test
    y_test = args.y_test
    if x_test != None:
        x_test = ''.join(x_test)
        y_test = ''.join(y_test)
        
        x_test = pd.read_csv(x_test).copy()
        y_test = pd.read_csv(y_test).copy()
        
        x_test = x_test.values[:,1:].astype(float)
        y_test = y_test.values[:,1:].astype(float)
        
        print('------------------------------------------------------------------------')
        out_test = mlp_model.forward(x_test)
        print(f'MSE error for test dataset:{mse_calculator(y_test, out_test)}%')
        print(f'number of error for test dataset:{error_calculator(y_test, out_test, 0.5)}')
        
    
    plt.subplot(2,1,1)
    plt.title('batchs error')
    plt.xlabel('n batch')
    plt.ylabel('ERROR')
    plt.plot(result[1])
    
    plt.subplot(2,1,2)
    plt.title('MSE error')
    plt.xlabel('n iter')
    plt.ylabel('ERROR (0-1)')
    plt.plot(result[0])
    
    plt.show()


if n_list[-1] != output_shape:
    raise ValueError('output shape and last neuron do not same shape...!')

print('------------------------------------------------------------------------')
print('Done')


