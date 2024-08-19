import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        bandwidths = self.get_bandwidth(L2_distances) * self.bandwidth_multipliers[:, None, None]
        bandwidths = bandwidths.expand(-1, L2_distances.size(0), L2_distances.size(1))
        return torch.exp(-L2_distances[None, ...] / bandwidths).sum(dim=0)

# class MMDLoss(nn.Module):
#     def __init__(self, kernel=RBF()):
#         super().__init__()
#         self.kernel = kernel

#     def forward(self, X, Y):
#         # Ensure both tensors are 2D and have compatible feature dimensions
#         X, Y = ensure_compatible_shapes(X, Y)
#         K = self.kernel(torch.vstack([X, Y]))
#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY
    
class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF(), batch_size=2000):
        super().__init__()
        self.kernel = kernel
        self.batch_size = batch_size

    def forward(self, X, Y):
        # Ensure both tensors are 2D and have compatible feature dimensions
        X, Y = ensure_compatible_shapes(X, Y)
        
        # Initialize accumulation for mean values
        XX = 0
        XY = 0
        YY = 0
        total_batches = (X.size(0) + self.batch_size - 1) // self.batch_size


        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, X.size(0))
            X_batch = X[start_idx:end_idx]
            Y_batch = Y[start_idx:end_idx]

            K = self.kernel(torch.vstack([X_batch, Y_batch]))
            X_size = X_batch.shape[0]
            XX += K[:X_size, :X_size].mean()
            XY += K[:X_size, X_size:].mean()
            YY += K[X_size:, X_size:].mean()

        # Average the results from all batches
        return XX / total_batches - 2 * XY / total_batches + YY / total_batches


def ensure_compatible_shapes(X, Y):
    # Flatten both tensors to 2D (batch_size, num_features)
    X_flat = X.view(X.size(0), -1)
    Y_flat = Y.view(Y.size(0), -1)

    # Pad the smaller tensor with zeros to match the number of features
    max_features = max(X_flat.size(1), Y_flat.size(1))
    if X_flat.size(1) < max_features:
        padding = torch.zeros(X_flat.size(0), max_features - X_flat.size(1), device=X.device)
        X_flat = torch.cat([X_flat, padding], dim=1)
    if Y_flat.size(1) < max_features:
        padding = torch.zeros(Y_flat.size(0), max_features - Y_flat.size(1), device=Y.device)
        Y_flat = torch.cat([Y_flat, padding], dim=1)

    return X_flat, Y_flat


# We also need to ensure that the tensors passed to MMDLoss in mmd_matrix function are handled properly:
def mmd_matrix(activations_1, activations_2):
    num_layers_1 = len(activations_1)
    num_layers_2 = len(activations_2)
    mmd_values = np.zeros((num_layers_1, num_layers_2))
    mmd_loss = MMDLoss()  # Initialize once and use for all computations

    for i in range(num_layers_1):
        for j in range(num_layers_2):
            X = torch.tensor(activations_1[i], dtype=torch.float32)
            Y = torch.tensor(activations_2[j], dtype=torch.float32)
            mmd_value = mmd_loss(X, Y).item()  # Compute MMD loss and convert to scalar
            mmd_values[i, j] = mmd_value

    return mmd_values

def load_activations(directory):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
    return [np.load(f) for f in files]

def main(folder1_path, folder2_path):
    folder1_files = os.listdir(folder1_path)
    folder2_files = os.listdir(folder2_path)

    # Sort the files to ensure consistent pairing
    folder1_files.sort()
    folder2_files.sort()
    activations_1 = load_activations(folder1_path)
    activations_2 = load_activations(folder2_path)

    mmd_values = mmd_matrix(activations_1, activations_2)

    return mmd_values

    plt.figure(figsize=(10, 8))
    sns.heatmap(mmd_values, annot=True, cmap='coolwarm', xticklabels=folder2_files, yticklabels=folder1_files)
    plt.title("Layer-wise MMD Similarity")
    plt.xlabel("Layers in Folder 2")
    plt.ylabel("Layers in Folder 1")
    plt.show()
    plt.savefig('mmd_testconv_ndp_epoch_100.png')

if __name__ == '__main__':
    folder1_path = "ndp_conv_activations_for_epoch_100"
    folder2_path = "dp_conv_activations_for_epoch_100"

    mmd_vals = main(folder1_path, folder2_path)

    print(mmd_vals)

    mmd_out = np.diag(mmd_vals).tolist()

    with open('mmd_epoch_100.json', 'w') as json_file:
        json.dump(mmd_out, json_file)


