import torch.nn as nn
class multilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation_layer, output_activation_layer, struct):
        super(multilayerPerceptron, self).__init__()
        
        layers = []
        
        if struct == 0:
            # All hidden layers have the same number of neurons (hidden_dim)
            dims = [input_dim] + [hidden_dim]*num_hidden_layers + [output_dim]
        
        elif struct == 1:
            # Bottleneck architecture
            dims = [input_dim]
            middle_layer_num = num_hidden_layers // 2
            middle_layer_neurons = round(hidden_dim * 2.5)
            increment = round((middle_layer_neurons - hidden_dim) / middle_layer_num)
            
            for _ in range(middle_layer_num):
                dims.append(dims[-1] + increment)
            
            dims.append(middle_layer_neurons)
            reversed_dims = dims[::-1]
            reversed_dims[0] = output_dim
            dims.extend(reversed_dims[1:])
        
        elif struct == 2:
            # Incremental Increase in Neurons
            dims = [input_dim]
            increment = round(hidden_dim*.25)
            for i in range(num_hidden_layers):
                dims.append(dims[-1] + increment)
            dims.append(output_dim)
        
        elif struct == 3:
            # Pyramid Architecture
            dims = [input_dim]
            decrement = round((input_dim - hidden_dim) / num_hidden_layers)
            for i in range(num_hidden_layers):
                dims.append(max(dims[-1] - decrement, hidden_dim))
            dims.append(output_dim)


        # Building the layers
        for i in range(len(dims) - 1):
            if i == len(dims) - 2:
                layers.append(nn.Linear(dims[i], output_dim))
            else:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2: 
                if activation_layer is not None:
                    layers.append(activation_layer)
        
        if output_activation_layer is not None:
            layers.append(output_activation_layer)
        
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)