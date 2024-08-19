import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

# BasicBlock definition
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(16, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(16, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(16, self.expansion*out_channels)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    


# Updating the RevisedResNet class with adaptive average pooling
class ConvNet(nn.Module):
    def __init__(self, block, depth, width_multiplier,num_groups):
        super(ConvNet, self).__init__()
        self.in_channels = 16 * width_multiplier

        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, self.in_channels)

        # Creating layers with increasing channels
        layers = []
        current_size = 32  # Assuming the input size is 32x32
        for i in range(1, depth + 1):
            out_channels = 16 * i * width_multiplier
            layers.append(block(self.in_channels, out_channels, stride=1))
            
            # Conditionally apply MaxPooling
            if current_size > 4:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size = current_size // 2
            
            self.in_channels = out_channels


        self.layers = nn.Sequential(*layers)

        # Adaptive average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.linear = nn.Linear(self.in_channels, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layers(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




