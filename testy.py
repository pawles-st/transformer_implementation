import torch.nn as nn
import torch
import sys 
from encoder import Encoder
import timeit
from classifier import Classifier
from dataset_loader import load_ag_news

def show_param(mod):
    for name, param in mod.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
    p = sum([p.numel() for p in mod.parameters()])
    print(f"Total number of parameters: {p}")

if __name__=="__main__":

    # ---------------------------------------
    # test encodera

    enc = Encoder(6, 512, 512, 512, 2048, 8)
    
    # sprawdzenie parametrów/warstw/wymiarów
    # show_param(enc)

    # sprawdzenie czy forward w encoderze działa 
    x_test = torch.randn(16, 512)
    print(enc(x_test))
    print(enc(x_test).shape) 


    # ---------------------------------------
    # test forward w klasyfikatorze

    model = Classifier(6, 512, 512, 512, 2048, 8)
    show_param(model)    

    # przykładowe dane 
    train_data, _ = load_ag_news(max_train_samples=5)
    print(f"Loaded {len(train_data)} examples from dataset.")
    text, label = train_data[0]
    print(text)

    # test czasu forward w encoderze
    N = 100
    t = timeit.timeit(lambda: model(text), number=N)
    print(t/N) # około 0.016