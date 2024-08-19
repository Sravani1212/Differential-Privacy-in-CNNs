import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming generate_tensors function is defined elsewhere
from tensor_gen import generate_tensors

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
        return torch.exp(-L2_distances[None, ...] / bandwidths).sum(dim=0)

class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def main():
    tensors = generate_tensors(10, 5, 0.1)  # Example call, adjust as necessary for your actual tensor generating function

    similarity_matrix = np.zeros((len(tensors), len(tensors)))
    for i, tensor_i in enumerate(tensors):
        for j, tensor_j in enumerate(tensors):
            # Convert tensors from NumPy arrays to PyTorch tensors
            Xi = torch.tensor(tensor_i.reshape(-1, 1), dtype=torch.float)
            Xj = torch.tensor(tensor_j.reshape(-1, 1), dtype=torch.float)

            # Compute MMD loss between each pair of tensors
            mmd_loss = MMDLoss()
            similarity = mmd_loss(Xi, Xj).item()  # Convert PyTorch tensor to a Python scalar

            similarity_matrix[i, j] = similarity

    # Visualize the similarity matrix in a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', vmin=np.min(similarity_matrix), vmax=np.max(similarity_matrix))
    plt.title("Tensor Similarity Heatmap Using MMD")
    plt.xlabel("Tensors")
    plt.ylabel("Tensors")
    plt.xticks(np.arange(len(tensors)), [f'T{i}' for i in range(len(tensors))])
    plt.yticks(np.arange(len(tensors)), [f'T{i}' for i in range(len(tensors))])
    plt.gca().invert_yaxis()
    plt.savefig("mmdonly.png")

if __name__ == "__main__":
    main()
