import torch
from sklearn.metrics import accuracy_score  # For calculating accuracy
import copy
import numpy as np

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs, task_type, verbose=True):

    training_results = {}
    
    epoch_count = []
    loss_values = []
    test_loss_values = []
    accuracy_values = []  # To store accuracy for classification tasks

    print(f"\n* Training Started *")
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        # Training Loop
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.float(), batch_y.float()  # Ensure the data is float type
            
            outputs = model(batch_X)
 
            if outputs.shape != batch_y.shape:
                outputs = outputs.view_as(batch_y)

            if task_type == 'mcc':
                loss = criterion(outputs, batch_y.long())
            elif task_type == 'bc':
                loss = criterion(outputs, batch_y.float())
            else:  
                loss = criterion(outputs, batch_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  # Set the model to evaluation mode
        # Evaluation Loop
        total_samples = 0
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.inference_mode():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.float(), batch_y.float()  # Ensure the data is float type
                outputs = model(batch_X)

                if outputs.shape != batch_y.shape:
                    outputs = outputs.view_as(batch_y)

                all_preds.extend(outputs.detach().numpy())
                all_labels.extend(batch_y.detach().numpy())

                if task_type == 'mcc':
                    loss = criterion(outputs, batch_y.long())
                elif task_type == 'bc':
                    loss = criterion(outputs, batch_y.float())
                else:  
                    loss = criterion(outputs, batch_y.float())
                test_loss += loss.item() * batch_y.shape[0]  # Accumulate total loss, weighted by batch size
                total_samples += batch_y.shape[0]  # Accumulate the total number of samples

        if len(test_loader) > 0:
            test_loss /= total_samples
        else:
            test_loss = 'N/A'

        # Calculate accuracy if it's a classification problem
        if task_type in ['mcc', 'bc']:
            if task_type == 'mcc':
                preds = np.argmax(all_preds, axis=1)
            else:
                all_preds_tensor = torch.FloatTensor(all_preds)
                preds = torch.round(torch.sigmoid(all_preds_tensor))
            accuracy = accuracy_score(all_labels, preds)
            accuracy_values.append(accuracy)

        if verbose:
            if epoch % (round(epochs*.1)) == 0:
                print_str = f"Epoch [{epoch+1}/{epochs}] | Training Loss: {loss.item():.8f} | Test Loss: {test_loss:.8f} |"
                if task_type in ['mcc', 'bc']:
                    print_str += f", Test Accuracy: {accuracy * 100:.2f}%"
                print(print_str)

        if epoch % (round(epochs*.5)) == 0:
            loss_values.append(loss.item())
            test_loss_values.append(test_loss)
            epoch_count.append(epoch+1)

        
    training_results['loss'] = loss_values
    training_results['test_loss'] = test_loss_values
    training_results['epochs'] = epoch_count
    if task_type in ['mcc', 'bc']:
        training_results['accuracy'] = accuracy_values

    print(f"* Training Finished *\n")
    return model, training_results
