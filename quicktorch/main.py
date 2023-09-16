from .mlp import mlp
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class quicktorch():
    def create_network(self, model_type, **kwargs):
        """
        Create a neural network model based on the specified model type and additional parameters.

        Args:
            model_type (str): The type of model to create. Current options include 'mlp'.
            **kwargs: Additional keyword arguments that vary based on the model type.

        For MLP (**kwargs can include):
            - task_type (str): The type of task to use. Current options include 'regression - (reg)', 'multiclass_classification - (mcc)', and 'binary_classification - (bc)'.
            - input_dim (int): The dimension of the input layer.
            - hidden_dim (int): The dimension of the hidden layers.
            - output_dim (int): The dimension of the output layer.
            - num_hidden_layers (int): The number of hidden layers.
            - optimizer_type (str): The type of optimizer to use ('SGD', 'Adam', etc.)
            - loss_function (str): The loss function to use ('MSE', 'CrossEntropy', etc.)
            - learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
            - hidden_activation_layer (str, optional): The activation function to use in the hidden layers ('relu', 'sigmoid', etc.). Default is None.
            - output_activation_layer (str, optional): The activation function to use in the output layer ('softmax', 'sigmoid', etc.). Default is None.
            - softmax_dim (int, optional): The dimension for the softmax function if used. Default is 1.
                - Typically used for classification tasks.
                - For a 2D input tensor `[batch_size, features]`, `softmax_dim` should be set to 1 to apply softmax across the feature dimension.
                - For a 3D input tensor `[batch_size, sequence_length, features]`, you might set `softmax_dim` to 2 to 
                apply softmax across the feature dimension for each sequence element.

        Returns:
            A neural network model of the specified type, initialized with the provided parameters.
        
        Example:
            >>> mlp_config ={
            >>> 'task_type': 'reg',
            >>> 'input_dim': 120,
            >>> 'hidden_dim': 480,
            >>> 'output_dim': 1,
            >>> 'num_hidden_layers': 2,
            >>> 'hidden_activation_layer':'relu',
            >>> }
            >>> mlp = quick.create_network('mlp', **mlp_config)
        """
        model_type = model_type.lower().strip()
        if model_type == 'mlp':
            return mlp(**kwargs)
        else:
            raise ValueError('Invalid model type: {}'.format(model_type))
        
    def data_prep(self, X, y, batch_size, test_size= 0.2, shuffle=True):
        """
        Split the input data into training and test sets.

        Args:
            X (np.ndarray or torch.Tensor): The input data.
            y (np.ndarray or torch.Tensor): The output data.
            batch_size (int): The batch size to use for training.
            test_size (float, optional): The fraction of the data to use for testing. Default is 0.2.
            shuffle (bool, optional): Whether to shuffle the data for training. Default is True for train. Test shuffle is always False.

        Returns:
            A tuple of training and test data loaders.
        """

        if isinstance(X, np.ndarray):
            X = torch.tensor(X.astype(np.float32))
        if isinstance(y, np.ndarray):
            y = torch.tensor(y.astype(np.float32))

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        
        return DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle), DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)