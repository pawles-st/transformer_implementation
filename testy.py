import torch.nn as nn
import torch
import sys 
from encoder import Encoder
import timeit
from classifier import Classifier
from dataset_loader import load_reviews

def show_param(mod):
    for name, param in mod.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
    p = sum([p.numel() for p in mod.parameters()])
    print(f"Total number of parameters: {p}")

if __name__=="__main__":

    # ---------------------------------------
    # test forward w klasyfikatorze

    model = Classifier(6, 512, 512, 512, 2048, 8)
    show_param(model)    

    # przykładowe dane 
    dataset = load_reviews()
    print(f"Loaded {len(dataset)} examples from dataset.")
    text, label = dataset[0]
    print(text)

    # test czasu forward w encoderze
    N = 100
    t = timeit.timeit(lambda: model(text), number=N)
    print(t/N) # około 0.016