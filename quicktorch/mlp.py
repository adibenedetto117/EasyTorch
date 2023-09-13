from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

class mlp():
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, optimizer_type, loss_function, learning_rate = 0.01, hidden_activation_layer = None, output_activation_layer = None, softmax_dim = 1):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation_layer = hidden_activation_layer_fn(hidden_activation_layer)
        self.output_activation_layer = output_activation_layer_fn(output_activation_layer, softmax_dim)

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        self.model = multilayerPerceptron(self.input_dim, self.hidden_dim, self.output_dim, self.num_hidden_layers, self.activation_layer, self.output_activation_layer)
    
    def __str__(self):
        return str(self.model)

class multilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation_layer, output_activation_layer):
        super(multilayerPerceptron, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))

        for i in range(num_hidden_layers):
            if activation_layer is not None:
                layers.append(activation_layer)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        if activation_layer is not None:
                layers.append(activation_layer)
        layers.append(nn.Linear(hidden_dim,output_dim))
        if output_activation_layer is not None or isinstance(output_activation_layer, nn.Linear):
            layers.append(output_activation_layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def hidden_activation_layer_fn(x):
    if x == None:
        return None
    elif x.lower() == 'relu':
        return nn.ReLU()
    elif x.lower() =='sigmoid':
        return nn.Sigmoid()
    elif x.lower() == 'tanh':
        return nn.Tanh()
    elif x.lower() == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError('Invalid hidden activation layer')
    
def output_activation_layer_fn(x, softmax_dim = 1):
    if x == None:
        return None
    elif x.lower() == "softmax":
        return nn.Softmax(dim=softmax_dim)
    elif x.lower() == "sigmoid":
        return nn.Sigmoid()
    elif x.lower() == "linear":
        return None
    elif x.lower() == "softsign":
        return nn.Softsign()
    else:
        raise ValueError('Invalid ouput activation layer')