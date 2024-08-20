import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import torch


def centering(K):
    n = K.shape[0]
    ones = np.ones(n)
    K_sum = np.dot(K, ones)  # Sum of each row in K
    total_sum = np.sum(K_sum)  # Sum of all elements in K
    centered_K = K - np.outer(K_sum, ones)/n - np.outer(ones, K_sum)/n + total_sum/(n**2)
    return centered_K

def linear_kernel(X, Y):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)  # Flatten all dimensions except the first
    if Y.ndim > 2:
        Y = Y.reshape(Y.shape[0], -1)  # Flatten all dimensions except the first

    return np.dot(X, X.T), np.dot(Y, Y.T)

def euclidean_dist_matrix(X, Y):
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(-1, 1)
    dist = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
    return np.abs(dist)

def rbf_kernel(X, Y, sigma_frac=0.4):
    dist_X = euclidean_dist_matrix(X, X)
    dist_Y = euclidean_dist_matrix(Y, Y)
    sigma_x = sigma_frac * np.percentile(dist_X, 0.5)
    sigma_y = sigma_frac * np.percentile(dist_Y, 0.5)
    K = np.exp(-dist_X / (2 * sigma_x ** 2))
    L = np.exp(-dist_Y / (2 * sigma_y ** 2))
    return K, L

def HSIC(K, L):
    m = K.shape[0]
    H = np.eye(m) - 1 / m * np.ones((m, m))
    HSIC_value = np.trace(np.dot(np.dot(np.dot(K, H), L), H)) / (m - 1) ** 2
    return HSIC_value

def CKA(X, Y, kernel_type='linear', sigma_frac=0.4):
    print(X.shape)
    print(Y.shape)
    if kernel_type == 'linear':
        K, L = linear_kernel(X, Y)
    elif kernel_type == 'rbf':
        K, L = rbf_kernel(X, Y, sigma_frac)
    centered_K = centering(K)
    centered_L = centering(L)
    return HSIC(centered_K, centered_L) / np.sqrt(HSIC(centered_K, centered_K) * HSIC(centered_L, centered_L))



def get_cka_matrix(activations_1, activations_2, kernel='linear', sigma_frac=0.4):
    num_layers_1 = len(activations_1)
    num_layers_2 = len(activations_2)
    cka_matrix = np.zeros((num_layers_1, num_layers_2))
    symmetric = num_layers_1 == num_layers_2

    for i in range(num_layers_1):
        for j in range(num_layers_2):
            if symmetric and j < i:
                cka_matrix[i, j] = cka_matrix[j, i]
            else:
                X, Y = activations_1[i], activations_2[j]
                cka_matrix[i, j] = CKA(X, Y, kernel, sigma_frac)

    return cka_matrix

def get_cka_matrix_batched(activations_1, activations_2, kernel='linear', sigma_frac=0.4, batch_size=256):
    num_layers_1 = len(activations_1)
    num_layers_2 = len(activations_2)
    cka_matrix = np.zeros((num_layers_1, num_layers_2))

    for i in range(num_layers_1):
        for j in range(num_layers_2):
            total_cka = 0
            num_batches = 0

            max_batches = min((activations_1[i].shape[0] - 1) // batch_size + 1, 
                              (activations_2[j].shape[0] - 1) // batch_size + 1)

            # Iterate over batches
            for batch_num in range(max_batches):
                batch_start = batch_num * batch_size
                batch_end = batch_start + batch_size

                X_batch = activations_1[i][batch_start:batch_end]
                Y_batch = activations_2[j][batch_start:batch_end]

                # Match the batch sizes by trimming the larger batch if necessary
                min_batch_size = min(X_batch.shape[0], Y_batch.shape[0])
                X_batch = X_batch[:min_batch_size]
                Y_batch = Y_batch[:min_batch_size]

                # Ensure the batch is non-empty and sizes match
                if X_batch.size > 0 and Y_batch.size > 0:
                    total_cka += CKA(X_batch, Y_batch, kernel, sigma_frac)
                    num_batches += 1

            if num_batches > 0:
                cka_matrix[i, j] = total_cka / num_batches

    return cka_matrix


def main(folder1_path, folder2_path, kernel_type='linear', sigma_frac=0.4):
    folder1_files = os.listdir(folder1_path)
    folder2_files = os.listdir(folder2_path)

    # Sort the files to ensure consistent pairing
    folder1_files.sort()
    folder2_files.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    activations_1 = [np.load(os.path.join(folder1_path, f), mmap_mode='r') for f in folder1_files]
    activations_2 = [np.load(os.path.join(folder2_path, f), mmap_mode='r') for f in folder2_files]


    cka_matrix = get_cka_matrix_batched(activations_1, activations_2, kernel_type, sigma_frac)

    # Save or process the CKA matrix as needed
    #print(cka_matrix)
    np.savetxt("cka_matrix.csv", cka_matrix, delimiter=",")
        # Generate and save a heatmap from the CKA matrix
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cka_matrix, annot=True, cmap='coolwarm', xticklabels=folder2_files, yticklabels=folder1_files)
    plt.title(f'CKA Similarity Heatmap ({kernel_type} kernel)')
    plt.xlabel('Layers in Folder 2')
    plt.ylabel('Layers in Folder 1')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cka_dpvndp_heatmap_conv_epoch_1.png")
    plt.show()

if __name__ == '__main__':
    folder1_path = " "
    folder2_path = " "

    main(folder1_path, folder2_path)
