import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def _centering(kernel):
    return kernel - kernel.mean(0, keepdims=True) - kernel.mean(1, keepdims=True) + kernel.mean()

def nbs(kernel1, kernel2):
    print(kernel1.shape)
    print(kernel2.shape)
    kernel1 = _centering(kernel1)
    kernel2 = _centering(kernel2)
    s, _ = torch.linalg.eig(kernel1 @ kernel2)
    s = s.real.clamp(0.).sqrt()
    return s.sum() / (torch.sqrt(kernel1.trace() * kernel2.trace()))


def get_nbs_matrix(activations_1, activations_2):
    num_layers_1 = len(activations_1)
    num_layers_2 = len(activations_2)
    nbs_matrix = np.zeros((num_layers_1, num_layers_2))
    for i in range(num_layers_1):
            for j in range(num_layers_2):
                X = activations_1[i]
                Y = activations_2[j]
                
                # Flatten the tensors if they are not already 2-dimensional
                if X.ndim > 2:
                    X = X.reshape(X.shape[0], -1)
                if Y.ndim > 2:
                    Y = Y.reshape(Y.shape[0], -1)

                tensor_X = torch.tensor(X, dtype=torch.float32)
                tensor_Y = torch.tensor(Y, dtype=torch.float32)

                kernel1 = tensor_X @ tensor_X.T
                kernel2 = tensor_Y @ tensor_Y.T

                nbs_matrix[i, j] = nbs(kernel1, kernel2)

    return nbs_matrix


def main(folder1_path, folder2_path):
    folder1_files = os.listdir(folder1_path)
    folder2_files = os.listdir(folder2_path)

    # Sort the files to ensure consistent pairing
    folder1_files.sort()
    folder2_files.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activations_1 = [np.load(os.path.join(folder1_path, f)) for f in folder1_files]
    activations_2 = [np.load(os.path.join(folder2_path, f)) for f in folder2_files]

    # Compute the NBS matrix
    nbs_matrix = get_nbs_matrix(activations_1, activations_2)

    # Plot the heatmap for the NBS matrix
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(nbs_matrix, annot=True, cmap='coolwarm', xticklabels=folder2_files, yticklabels=folder1_files)
    plt.title(f'NBS Similarity Heatmap')
    plt.xlabel('Layers in Folder 2')
    plt.ylabel('Layers in Folder 1')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig("nbs_heatmap_epoch_1.png")
    plt.show()

if __name__ == '__main__':
    folder1_path = "dp_conv_activations_for_epoch_1"
    folder2_path = "dp_conv_activations_for_epoch_1"

    main(folder1_path, folder2_path)
