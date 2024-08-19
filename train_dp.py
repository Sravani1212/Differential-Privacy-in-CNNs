import hydra
from dataclasses import dataclass,field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import hydra.utils
import torch
import torch.nn as nn
import torch.optim as optim
from utils_dp import setup_privacy_engine
from utils import get_activations,load_data
import warnings 
from opacus.validators import ModuleValidator
import importlib
import matplotlib.pyplot as plt
from convnet import BasicBlock 
#from resnet import BasicBlock
import logging
logging.getLogger('opacus').setLevel(logging.WARNING)


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

    #print("Current Working Directory:", os.getcwd())
    model_module = importlib.import_module(cfg.model.name.lower())
    model_class = getattr(model_module, cfg.model.name)

    # Instantiate the model with unpacked parameters
    model_params = {**cfg.model.params,'block':BasicBlock}
    model = model_class(**model_params)


    train_loader, testloader = load_data(batch_size)

    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,weight_decay=0.09)
    elif cfg.optimizer == 'rms':
        optimizer = optim.rmsprop(model.parameters(), lr=cfg.learning_rate,alpha=0.99,weight_decay=0.9)



    warnings.simplefilter("ignore")
    MAX_GRAD_NORM = 1.0
    target_epsilon = 1.0
    NOISE_MULTIPLIER = 10
    target_delta = 1e-5
    
    model, optimizer, train_loader, privacy_engine = setup_privacy_engine(
        model, 
        optimizer,  
        train_loader,
        num_epochs,
        target_epsilon, 
        target_delta, 
        MAX_GRAD_NORM
    )

    criterion = nn.CrossEntropyLoss()
        # Lists to store training accuracy and loss, and testing accuracy and loss for plotting
    train_accuracy_values = []
    train_loss_values = []
    test_accuracy_values = []
    test_loss_values = []


    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()  # Set the model to training mode
        
        for i, data in enumerate(train_loader, 0):
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

        train_accuracy = 100 * correct / total
        train_accuracy_values.append(train_accuracy)
        train_loss_values.append(running_loss / len(train_loader))
        
        # Print the training accuracy and loss at the end of each epoch
        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}, Train Accuracy: {100 * correct / total}%")
        
    print("Finished Initial Training")
        

    # Save the model
    torch.save(model.state_dict(), 'dp_model_con1_weights.pth')
    print("Model Saved")

    # Create a new instance of the model
    loaded_model = model_class(**model_params)
    #loaded_model.load_state_dict(torch.load('dp_model_con1_weights.pth'))

    

    MAX_GRAD_NORM = 1.0
    target_epsilon = 1.0
    NOISE_MULTIPLIER = 10
    target_delta = 1e-5


    #loaded_model = model_class(**model_params)
    loaded_model = ModuleValidator.fix(loaded_model)
    ModuleValidator.validate(loaded_model, strict=False)  


    # Save the model weights and biases to a pickle file
    weights_path = 'dp_model_con1_weights.pth'
    # Save the model with removed "_module" prefix from state_dict keys
    torch.save(model.state_dict(), 'dp_model_con1_weights.pth')


    try:
        # Load the state dictionary
        state_dict = torch.load(weights_path)

        # Remove the "_module" prefix from keys (if present)
        state_dict = {key.replace("_module.", ""): value for key, value in state_dict.items()}

        # Load the modified state dictionary into the model
        loaded_model.load_state_dict(state_dict)

        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Model weights file '{weights_path}' not found.")
    except RuntimeError as e:
        print(f"Error loading model weights: {str(e)}")

        # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(loaded_model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(loaded_model.parameters(), lr=cfg.learning_rate,weight_decay=0.09)
    elif cfg.optimizer == 'rms':
        optimizer = optim.rmsprop(loaded_model.parameters(), lr=cfg.learning_rate,alpha=0.99,weight_decay=0.9)


    for epoch in range(num_epochs):
        test_running_loss = 0.0
        correct = 0
        total = 0
        loaded_model.eval()

        if epoch+1 in cfg.epochs_for_activations:
            save_directory = f'dp_conv_activations_for_epoch_{epoch+1}'
            get_activations(model, train_loader, epoch+1, save_directory)

        with torch.no_grad():
            for i,data in enumerate(testloader,0):
                inputs, labels = data
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
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
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(train_accuracy_values, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(train_loss_values, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(test_accuracy_values, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(test_loss_values, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.savefig('convnet_dp.png')  # Save the plots to a file



if __name__ == '__main__':
    train()


