import torch.nn as nn
import torch
import sys 
from encoder import Encoder


def show_param(mod):
    for name, param in mod.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")

enc = Encoder(6, 512, 512, 512, 2048, 8)
show_param(enc)

x_test = torch.randn(10, 16, 512)
print(enc(x_test[0]))
print(enc(x_test[0]).shape)