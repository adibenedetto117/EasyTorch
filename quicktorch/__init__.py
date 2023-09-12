#from .cnn import CNN
#from .rnn import YourRNNClass
#from .lstm import YourLSTMClass
from fnn import fnn

class quicktorch():
    def create_network(self, model_type, **kwargs):
        model_type = model_type.lower().strip()
        if model_type == 'fnn':
            return fnn(**kwargs)
       
nnFactory = quicktorch()
x = nnFactory.create_network('fnn', input_dim=10, hidden_dim=40, output_dim=1, num_hidden_layers = 10, 
                           optimizer_type ="Adam", loss_function="mae")
print(x)