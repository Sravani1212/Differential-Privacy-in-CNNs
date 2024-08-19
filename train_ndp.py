import hydra
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import hydra.utils
import torch
import importlib
import torch.nn as nn
from convnet import BasicBlock 
#from resnet import BasicBlock
import torch.optim as optim
import os
from utils import get_activations, load_data
import matplotlib.pyplot as plt
import numpy as np


@ dataclass
# Define a configuration schema
class ModelConfig:
    name: str 
    params: dict 

@ dataclass
class Config:
    model: ModelConfig 
    optimizer: str = "sgd"
    epochs_for_activations: list = field(default_factory=list)
    learning_rate: float = 0.001
    num_epochs: int = 3
    batch_size: int = 64
    iterations: int = 1000
    target_iter: list= field(default_factory=list)
    

# Register the configuration schema
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_path=".", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Script started")
        # Now you can use cfg to access your configuration variables
    optimizer = cfg.optimizer
    
    learning_rate = cfg.learning_rate
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    
    model_module = importlib.import_module(cfg.model.name.lower())
    model_class = getattr(model_module, cfg.model.name)

    # Instantiate the model with unpacked parameters
    model_params = {**cfg.model.params,'block':BasicBlock}
    model = model_class(**model_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #print("Current Working Directory:", os.getcwd())
    trainloader, testloader = load_data(batch_size)


    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,weight_decay=0.09)
    elif cfg.optimizer == 'rms':
        optimizer = optim.rmsprop(model.parameters(), lr=cfg.learning_rate,alpha=0.99,weight_decay=0.9)


    num_epochs = cfg.num_epochs
    # Lists to store training accuracy and loss, and testing accuracy and loss for plotting
    train_accuracy_values = []
    train_loss_values = []
    test_accuracy_values = []
    test_loss_values = []

    # Initial training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()  # Set the model to training mode
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if epoch == 49 or epoch == 99:
            #     save_dir = f'weights_epoch_{epoch+1}'
            #     os.makedirs(save_dir, exist_ok=True)
            #     for name, parameter in model.named_parameters():
            #         if parameter is not None:
            #             sanitized_name = name.replace('.', '_')
            #             file_path = os.path.join(save_dir, f"{sanitized_name}weights_epoch{epoch+1}.npy")
            #             np.save(file_path, parameter.detach().cpu().numpy())
            #     print(f"Weights saved for epoch {epoch+1}")
            # save_dir = 'gradients_epoch_100'
            # os.makedirs(save_dir, exist_ok=True) 
            # if epoch == 99:
            #     for name, parameter in model.named_parameters():
            #         if parameter.grad is not None:
            #         # Replace dots in layer names with underscores to avoid file naming issues
            #             sanitized_name = name.replace('.', '_')
            #             file_path2 = os.path.join(save_dir, f"{sanitized_name}gradients_dp_iter{epoch+1}.npy")
            #         # Save the gradient as a NumPy array
            #             np.save(file_path2, parameter.grad.detach().cpu().numpy())

        train_accuracy = 100 * correct / total
        train_accuracy_values.append(train_accuracy)
        train_loss_values.append(running_loss / len(trainloader))
        
        # Print the training accuracy and loss at the end of each epoch
        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader)}, Train Accuracy: {100 * correct / total}%")
        


    print("Finished Initial Training")

    # Save the model weights and biases to a pickle file
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Finished Saving Model")

    # Load the model weights from the pickle file
    loaded_model = model_class(**model_params)
    loaded_model.load_state_dict(torch.load('model_weights.pth'))
    loaded_model = loaded_model.to(device)



    def accuracy(preds, labels):
        return (preds == labels).mean()


    # Further training using the loaded model
    for epoch in range(num_epochs):

        running_loss = 0.0
        correct = 0
        total = 0
        
        loaded_model.train()  # Set the model to training mode
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Evaluate the model on the test dataset
        correct = 0
        total = 0
        test_running_loss = 0.0
        
        loaded_model.eval()  # Set the loaded model to evaluation mode

        if epoch+1 in cfg.epochs_for_activations:
            save_directory = f'ndp_test_conv_activations_for_epoch_{epoch+1}'
            get_activations(loaded_model, trainloader, epoch+1, save_directory)


        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = loaded_model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                
                # Calculate test accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        test_accuracy_values.append(test_accuracy)
        test_loss_values.append(test_running_loss / len(testloader))
        
        # Print the test accuracy and loss at the end of each epoch
        print(f"Epoch {epoch+1}, Test Loss: {test_running_loss / len(testloader)}, Test Accuracy: {test_accuracy}%")

    print("Finished Training and Testing")

        # Plot training and test accuracy and loss
    # plt.figure(figsize=(12, 8))

    # plt.subplot(2, 2, 1)
    # plt.plot(train_accuracy_values, label='Train Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.grid()

    # plt.subplot(2, 2, 2)
    # plt.plot(train_loss_values, label='Train Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid()

    # plt.subplot(2, 2, 3)
    # plt.plot(test_accuracy_values, label='Test Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.grid()

    # plt.subplot(2, 2, 4)
    # plt.plot(test_loss_values, label='Test Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid()

    # plt.savefig('convnet_ndp.png')  

if __name__ == '__main__':
    train()
