from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

class CNN():
    """
    QTCNN: A wrapper for QuickCNN to ease the training and prediction processes.
    
    Parameters For Set
    ----------
    input_shape : tuple
        The shape of a single input example as (channels, height, width).
    task_type : str, optional
        The type of task to perform ("classification" or "regression"). Default is "classification".
    output_dim : int, optional
        Number of output dim for classification tasks. Default is 10.
    filters : list of int, optional
        List of filters for each Conv2D layer. Default is [32, 64].
    kernel_sizes : list of int, optional
        List of kernel sizes for each Conv2D layer. Default is [3, 3].
    strides : list of int, optional
        List of strides for each Conv2D layer. Default is [1, 1].
    padding : list of str, optional
        List of padding types ('same' or 'valid') for each Conv2D layer. Default is ['same', 'same'].
    activation : str, optional
        Type of activation function ('relu' or 'leaky_relu'). Default is 'relu'.
    batch_norm : bool, optional
        Whether to include Batch Normalization layers. Default is True.
    pooling_type : str, optional
        Type of pooling to be used ('max' or 'average'). Default is 'max'.
    pooling_size : int, optional
        Size of the pooling layer. Default is 2.
    dense_layers : list of int, optional
        List of neurons in each dense (fully-connected) layer. Default is [128].
    dropout : float, optional
        Dropout rate for dropout layers. Default is 0.5.
    optimizer : str, optional
        Type of optimizer ('adam' or 'sgd'). Default is 'adam'.
    learning_rate : float, optional
        Learning rate for optimizer. Default is 0.001.
    loss_function : str, optional
        Loss function for the model ('cross_entropy' or 'mse' or 'mae'). Default is 'cross_entropy'.

    Example:
    --------
    >>> input_shape = (3, 32, 32)  # Channels x Height x Width
    >>> num_classes = 10
    >>> my_qtcnn = QTCNN(input_shape, num_classes=num_classes)
    >>> print(my_qtcnn)
    """
    def __init__(self):
        
        self.train_state = False
        self.model = None
        
        
    def train(self, X_train, y_train, X_val, y_val, epochs):
        pass

    def predict(self, X):
        pass

    def set(self, input_shape, task_type = "classification", output_dim=10, filters=[32, 64], kernel_sizes=[3, 3], strides=[1, 1], padding=['same', 'same'], activation='relu', 
                 batch_norm=True, pooling_type='max', pooling_size=2, dense_layers=[128], dropout=0.5, optimizer='adam', learning_rate=0.001, 
                 loss_function='cross_entropy'):
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

        self.task_type = task_type

        if self.task_type == "classification":
            self.model = QuickCNN(input_shape, output_dim, filters, kernel_sizes, strides, padding, activation, batch_norm, pooling_type, pooling_size, dense_layers, dropout)
            if self.loss_function == 'cross_entropy':
                self.loss_function = nn.CrossEntropyLoss()
            else:
                self.loss_function = nn.CrossEntropyLoss()
        elif self.task_type == "regression":
            self.model = QuickCNN(input_shape, output_dim, filters, kernel_sizes, strides, padding, activation, batch_norm, pooling_type, pooling_size, dense_layers, dropout)
            if self.loss_function == 'mse':
                self.loss_function = nn.MSELoss()
            elif self.loss_function == 'mae':
                self.loss_function = nn.L1Loss()
            else:
                self.loss_function = nn.MSELoss()

    def __str__(self):
        return str(self.model)

    

class QuickCNN(nn.Module):
    """
    QuickCNN: A simplified Convolutional Neural Network (CNN) using PyTorch.
    
    This class provides a way to create a straightforward CNN with options for customizing
    various layers including Conv2D, Activation, BatchNorm, Pooling, Dense, and Dropout.
    
    Parameters:
    ----------
    input_shape : tuple
        Shape of a single input example (channels, height, width).
    task : str, optional
        The type of task ("classification" or "regression"). Default is "classification".
    output_dim : int
        Number of output dim for classification tasks.
    filters : list of int, optional
        List of integers specifying the number of filters for each Conv2D layer. (default: [32, 64])
    kernel_sizes : list of int, optional
        List of integers specifying the kernel sizes for each Conv2D layer. (default: [3, 3])
    strides : list of int, optional
        List of integers specifying the strides for each Conv2D layer. (default: [1, 1])
    padding : list of str, optional
        List of strings specifying padding type ('same' or 'valid') for each Conv2D layer. (default: ['same', 'same'])
    activation : str, optional
        Type of activation function ('relu' or 'leaky_relu'). (default: 'relu')
    batch_norm : bool, optional
        Whether to include Batch Normalization layers. (default: True)
    pooling_type : str, optional
        Type of pooling to be used ('max' or 'average'). (default: 'max')
    pooling_size : int, optional
        Size of the pooling layer. (default: 2)
    dense_layers : list of int, optional
        List of integers specifying the number of neurons in each dense (fully-connected) layer. (default: [128])
    dropout : float, optional
        Dropout rate for the dropout layers. (default: 0.5)
    
    Example:
    --------
    >>> input_shape = (3, 32, 32)
    >>> num_classes = 10
    >>> model = QuickCNN(input_shape, num_classes=num_classes)
    >>> print(model)
    """


    def __init__(self, input_shape, output_dim=None, filters=[32, 64], kernel_sizes=[3, 3], strides=[1, 1], padding=['same', 'valid'], activation='relu', batch_norm=True, pooling_type='max', pooling_size=2, dense_layers=[128], dropout=0.5):
        super(QuickCNN, self).__init__()
        
        # Basic validations
        if len(filters) != len(kernel_sizes) or len(filters) != len(strides) or len(filters) != len(padding):
            raise ValueError("Mismatch in the length of layer parameters")
        
        input_channels, _, _ = input_shape
        layers = []
        for i, (out_channels, kernel_size, stride, pad) in enumerate(zip(filters, kernel_sizes, strides, padding)):
            conv_pad = (kernel_size - 1) // 2 if pad == 'same' else 0
            
            layers.extend([
                nn.Conv2d(input_channels if i == 0 else filters[i-1], out_channels, kernel_size, stride, conv_pad),
                nn.ReLU() if activation == 'relu' else nn.LeakyReLU(),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.MaxPool2d(pooling_size) if pooling_type == 'max' else nn.AvgPool2d(pooling_size)
            ])
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the size of the flattened features to connect to the fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.numel()
        
        # Fully Connected layers
        fc_layers = []
        prev_layer_size = flattened_size
        for num_neurons in dense_layers:
            fc_layers.extend([
                nn.Linear(prev_layer_size, num_neurons),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_layer_size = num_neurons
        
        fc_layers.append(nn.Linear(prev_layer_size, output_dim))
        
        self.classifier = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten tensor for fully connected layer
        x = self.classifier(x)
        return x