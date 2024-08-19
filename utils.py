import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
from math import ceil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_activations(model, data_loader, epoch, save_dir):
    model.eval()  # Set the model to evaluation mode
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu().numpy())
        return hook

    # Register hooks to collect activations
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Process the data for the specified epoch
    for i, data in enumerate(data_loader, start=1):
        if i > epoch:
            break
        inputs, _ = data
        inputs=inputs.to(device)
        _=_.to(device)
        _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save the activations in separate files
    os.makedirs(save_dir, exist_ok=True)
    for name, acts in activations.items():
        filename = os.path.join(save_dir, f'{name}_epoch_{epoch}.npy')
        np.save(filename, np.concatenate(acts, axis=0))


    


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    batch_size = batch_size

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
