from typing import Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
from .mlparchitecture import multilayerPerceptron
from .functions import hidden_activation_layer_fn, output_activation_layer_fn, optimizer_fn, loss_fn
from .train import train_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error


class mlp():
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int,hidden_activation_layer: Optional[str] = None, 
                 output_activation_layer: Optional[str] = None, softmax_dim: int = 1, task_type: str = None, struct: int = 0, ):
        
        if task_type == None or task_type not in ["reg", "mcc", "bc"]:
            raise ValueError("Task type must be specified. Options for mlp are 'reg', 'mcc, 'bc'. View docs.")
        if struct not in [0, 1, 2, 3]:
            raise ValueError("struct must be 0, 1, 2, 3 | 1 = 'hourglass' | 2 = 'pyramid View docs.")
        
        self.task_type = task_type
        self._trained = False

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation_layer = hidden_activation_layer_fn(hidden_activation_layer)
        self.output_activation_layer = output_activation_layer_fn(output_activation_layer, softmax_dim)
        self.struct = struct

        self.learning_rate = None
        self.optimizer_type = None
        self.loss_function = None
        self.weight_decay = None
        self.momentum = None

        self.training_results = {}

        self.model = multilayerPerceptron(self.input_dim, self.hidden_dim, self.output_dim, self.num_hidden_layers, self.activation_layer, self.output_activation_layer, self.struct)

    def compile(self, optimizer_type: str, loss_function: str, learning_rate: float = 0.01, 
                            weight_decay: float = 0, momentum: float = 0):
        """
        Sets the hyperparameters for the model's optimizer and loss function.

        Parameters:
            optimizer_type (str): The type of optimizer to use. Supported types are 'adam', 'sgd', 'rmsprop', and 'adagrad'.
                                - 'adam': Uses the Adam optimizer. Relevant settings are `learning_rate` and `weight_decay`.
                                - 'sgd': Uses the Stochastic Gradient Descent optimizer. Relevant settings are `learning_rate`, `weight_decay`, and `momentum`.
                                - 'rmsprop': Uses the RMSprop optimizer. Relevant settings are `learning_rate` and `weight_decay`.
                                - 'adagrad': Uses the Adagrad optimizer. Relevant settings are `learning_rate` and `weight_decay`.
            loss_function (str): The loss function to use during training.
                - (str): The type of the loss function ('cross_entropy','mse','mae', ''BCEWithLogitsLoss').
            learning_rate (float, optional): The learning rate for the optimizer. Default is 0.01.
            weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Default is 0.
            momentum (float, optional): The momentum factor for SGD optimizer. Default is 0. Only relevant if `optimizer_type` is 'sgd'.

        Note:
            This function uses the `optimizer_fn` internally to create the optimizer based on the specified `optimizer_type` and settings.

        Returns:
            None
        """
        
        self.learning_rate = learning_rate

        self.optimizer_type = optimizer_fn(optimizer_type, self.model, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        self.loss_function = loss_fn(loss_function)

    def train(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, epochs: int, verbose = True, fig_size: tuple = (18,6)):
        """
        Trains a PyTorch model and evaluates its performance.
        
        Parameters:
            model (torch.nn.Module): The PyTorch model to train.
            train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
            test_loader (torch.utils.data.DataLoader): The DataLoader for the test data.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            criterion (torch.nn.Module): The loss function to use. 
                - For 'mcc' with a softmax activation, use nn.CrossEntropyLoss.
                - For 'bc' with a sigmoid activation, use nn.BCEWithLogitsLoss.
                - For 'reg', use nn.MSELoss or similar.
            epochs (int): The number of training epochs.
            task_type (str): The type of machine learning task. 
                            'mcc' for multi-class classification,
                            'bc' for binary classification, 
                            'reg' for regression.
            output_activation_layer (torch.nn.Module, optional): The output activation layer, if any. 
                - For 'mcc', typically use a softmax activation.
                - For 'bc', typically use a sigmoid activation.
                - For 'reg', typically no activation is applied. 
                (Default is None)
            verbose (bool): Whether to print training progress. (Default is True)
            
        Returns:
            tuple: A tuple containing the trained model and a dictionary of training results.
        
        Notes:
            - For 'mcc' and 'bc', the function also calculates and stores accuracy.
            - Make sure to use the appropriate loss function for your task to ensure correct training.
        """
        if self._trained == False:
            self.model, self.training_results = train_model(self.model, train_loader, test_loader, self.optimizer_type, 
            self.loss_function,output_activation_layer = self.output_activation_layer, epochs=epochs,task_type=self.task_type, verbose= verbose)
            self._trained = True
            if verbose:
                self.show_training_metrics(fig_size=fig_size)  
        else:
            raise ValueError("Model has already been trained.")
    
    def __str__(self):
        return str(self.model)
    
    def show_training_metrics(self, fig_size: tuple = (18,6)):
        fig = plt.figure(figsize=fig_size)
        if self._trained == True:
            plt.plot(self.training_results["epochs"], self.training_results['loss'], label="Training loss")
            plt.plot(self.training_results["epochs"], self.training_results['test_loss'], label="Testing loss")
            if self.task_type == "mcc" or self.task_type == "bc":
                plt.plot(self.training_results["epochs"], self.training_results['accuracy'], label="Testing accuracy")
            plt.legend()
            plt.show()
        else:
            raise ValueError("Model has not been trained.")
        
    def predict(self, test_loader: torch.utils.data.DataLoader):
        if self._trained:
            self.model.eval()  # set the model to evaluation mode
            
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch['inputs'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    if self.task_type in ["mcc", "bc"]:
                        # Get the predicted labels
                        _, predicted = torch.max(outputs, 1)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                    elif self.task_type == "reg":
                        all_predictions.extend(outputs.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
            
            if self.task_type in ["mcc", "bc"]:
                accuracy = accuracy_score(all_labels, all_predictions)
                print(f"Accuracy: {accuracy * 100:.2f}%")
                
            elif self.task_type == "reg":
                mse = mean_squared_error(all_labels, all_predictions)
                print(f"Mean Squared Error: {mse:.2f}")
                
            # Add any additional plots or visualizations here.
            if self.task_type in ["mcc", "bc"]:
                plt.hist(all_predictions, alpha=0.5, label='Predictions')
                plt.hist(all_labels, alpha=0.5, label='True Labels')
                plt.legend(loc='upper right')
                plt.show()
                
            return all_predictions
        else:
            raise ValueError("Model has not been trained.")
        

    

    def save_model(self, PATH: str, metadata: dict = None):
        try:
            save_object = {
                'model_state_dict': self.model.state_dict(),
                'metadata': metadata
            }
            torch.save(save_object, PATH)
            print(f"Model saved successfully at {PATH}")
            return "Model saved at {PATH}"

        except Exception as e:
            print(f"Error saving model: {e}")
            return f"Error saving model: {e}"

    def display_model(self):
        print(self.model)
    
    def display_hyperparameters(self):
        print("\n************** Hyperparameters **************")
        print(f'Input dimension: {self.input_dim}')
        print(f'Hidden dimension: {self.hidden_dim}')
        print(f'Output dimension: {self.output_dim}')
        print(f'Number of hidden layers: {self.num_hidden_layers}')
        print(f'Activation layer: {self.activation_layer}')
        print(f'Output activation layer: {self.output_activation_layer}')
        print(f'Loss function: {self.loss_function}')
        print(f'Optimizer type: {self.optimizer_type}')
        print("*********************************************\n")

    

    
