from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

class CNN():
    def __init__(self, input_shape, task_type = "classification", output_dim=10, filters=[32, 64], kernel_sizes=[3, 3], strides=[1, 1], padding=['same', 'same'], activation='relu', 
                 batch_norm=True, pooling_type='max', pooling_size=2, dense_layers=[128], dropout=0.5, optimizer='adam', learning_rate=0.001, 
                 loss_function='cross_entropy'):
        
        # Loss Functions
        loss_functions = {
        
        }
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.batch_norm = batch_norm
        self.pooling_type = pooling_type
        self.pooling_size = pooling_size
        self.dense_layers = dense_layers
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        self.model = CNN_Structure()

    def __str__(self):
        return str(self.model)

    

class CNN_Structure(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 82, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(82 * 37, 512)
        self.fc2 = nn.Linear(512, 120)
        self.fc3 = nn.Linear(120, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x