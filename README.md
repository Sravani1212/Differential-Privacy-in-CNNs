# Impact of Differential Privacy on Large Convolutional Neural Networks (CNNs)

This repository contains the implementation and analysis of the impact of differential privacy (DP) on large Convolutional Neural Networks (CNNs). We train a custom ConvNet model with and without differential privacy and evaluate the similarity between layers using various metrics such as Centered Kernel Alignment (CKA), Maximum Mean Discrepancy (MMD), and Neuron Activation-Based Similarity (NBS).

# Project Overview

# Model Architecture

The core of our ConvNet model is based on a modified version of ResNet20, designed with the following specifications:

- Network Structure: The model is constructed by stacking `BasicBlocks`. The depth and width of the network are configurable parameters. Each `BasicBlock` is responsible for a specific transformation, with the number of output channels increasing as we progress through the network.
  
- Depth and Width: The depth defines the number of `BasicBlocks` in the network, while the width controls the multiplier for the output channels. For example, with a depth of 4 and a width of 3, the network will have 4 `BasicBlocks`, and the output channels for the first block will be (16 * 1 * 3), the second will be (16 * 2 * 3), and so on.

- Convolutional Layers: The network begins with a standard convolutional layer, followed by a series of `BasicBlocks`. Instead of regular `Conv2d`, we use weight standardized convolutions for better training stability.

- Normalization: We replace every `BatchNorm2d` layer with `GroupNorm` using 16 groups, which offers more robust performance when training with differential privacy.

- Pooling: To reduce the size of the feature maps, a `MaxPooling` layer with a 2x2 kernel is placed after each `BasicBlock`.

# Training Procedure

We train the network with and without differential privacy (DP) using a depth of 4 and a width of 3. The training is conducted on a standard image classification dataset, and the model's performance is evaluated in terms of accuracy and privacy loss.

# Similarity Metrics

After training, we analyze the similarity between the layers of the network using the following metrics:

1. Centered Kernel Alignment (CKA): Measures the similarity between the representations of different layers.
2. Maximum Mean Discrepancy (MMD): Compares the distributions of features across layers.
3. Neuron Activation-Based Similarity (NBS): Assesses the similarity of activations across different layers.

# Repository Structure

- `models/`: Contains the implementation of the custom ConvNet model.
- `train.py`: Script to train the model with and without differential privacy.
- `data/`: Directory for storing datasets (not included in the repository for privacy reasons).

# Usage

# Training the Model

To train the model with default settings:

python train.py --depth 4 --width 3 --dp True

## Acknowledgements

- The implementation of weight standardized convolution is adapted from [this repository](https://github.com/joe-siyuan-qiao/pytorch-classification).
