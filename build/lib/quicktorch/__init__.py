from .cnn import CNN
#from .rnn import YourRNNClass
#from .lstm import YourLSTMClass

def create_model(model_type):
    """Creates a model based on the model type.
        CNN, RNN, or LSTM are supported.
        Args:
            model_type (str): Model type.
            'cnn'
            'rnn'
            'lstm'
    """
    model_type = model_type.lower().strip()
    if model_type == 'cnn':
        return CNN()
    #elif model_type == 'rnn':
        #return YourRNNClass()
    #elif model_type == 'lstm':
        #return YourLSTMClass()
    else:
        raise ValueError("Invalid model type.")