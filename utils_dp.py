import torch
import torch.nn as nn
import numpy as np
import os
from opacus import PrivacyEngine



def setup_privacy_engine(model, optimizer, train_loader,epochs, target_epsilon, target_delta, max_grad_norm):
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm
    )

    return model, optimizer, train_loader, privacy_engine

