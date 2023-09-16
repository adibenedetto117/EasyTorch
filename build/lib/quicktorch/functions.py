import torch.nn as nn
import torch.optim as optim

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
    
def optimizer_fn(x, model, **kwargs):
    """
    Factory function for creating PyTorch optimizers.

    Parameters:
        x (str): The type of the optimizer ('adam', 'sgd', 'rmsprop', 'adagrad').
        kwargs (dict): Additional keyword arguments for the optimizer constructor.

    Returns:
        torch.optim.Optimizer: A PyTorch optimizer instance.
    """
    try:
        if x is None:
            return None
        elif x.lower() == 'adam':
            relevant_args = {k: v for k, v in kwargs.items() if k in ['learning_rate', 'weight_decay']}
            return optim.Adam(**relevant_args, params=model.parameters())
        elif x.lower() == 'sgd':
            return optim.SGD(**kwargs, params=model.parameters())
        elif x.lower() == 'rmsprop':
            relevant_args = {k: v for k, v in kwargs.items() if k in ['learning_rate', 'weight_decay']}
            return optim.RMSprop(**relevant_args, params=model.parameters())
        elif x.lower() == 'adagrad':
            relevant_args = {k: v for k, v in kwargs.items() if k in ['learning_rate', 'weight_decay']}
            return optim.Adagrad(**relevant_args, params=model.parameters())
        else:
            raise ValueError('Invalid optimizer')
    except:
        raise ValueError('Invalid hyperparameters for current optimizer. Please check docs.')
    
def loss_fn(x, **kwargs):
    """
    Factory function for creating PyTorch loss functions.

    Parameters:
        x (str): The type of the loss function ('cross_entropy','mse','mae', 'bce_with_logits').
        kwargs (dict): Additional keyword arguments for the loss function constructor.

    Returns:
        torch.nn.Module: A PyTorch loss function instance.
    """
    try:
        if x is None:
            return None
        elif x.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss(**kwargs)
        elif x.lower() =='mse':
            return nn.MSELoss(**kwargs)
        elif x.lower() =='mae':
            return nn.L1Loss(**kwargs)
        elif x.lower() == 'BCEWithLogitsLoss'.lower():
            return nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError('Invalid loss function')
    except:
        raise ValueError('Invalid hyperparameters for current loss function. Please check docs.')