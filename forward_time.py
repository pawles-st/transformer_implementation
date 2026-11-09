import torch 
import torch.nn as nn
from classifier import Classifier
from dataset_loader import load_sst2
import sys 
import timeit


def forward_time(batch_size: int, N: int, heads: list[int], layers: list[int], d_e_list: list[int], d_k_list: list[int], dff_list: list[int]):
	# przykładowe dane 
    train_data, val_data = load_sst2()
    batches = [train_data[i*batch_size:(i+1)*batch_size] for i in range(N)]

    for l in layers:
    	for h in heads:
    		for de in d_e_list:
    			for dk in d_k_list:
    				for dff in dff_list:
    					model = Classifier(l, de, dk, dk, dff, h)
    					t = 0
    					for i in range(N):
    						text = [sentence for sentence, _ in batches[i]]
    						t += timeit.timeit(lambda: model(text), number=1)
    					print(f'Parametry modelu: nr_layers={l}, h={h}, d_e={de}, d_k=d_q={dk}, d_ff={dff}')
    					print(f'Średni czas batcha o rozmiarze {batch_size}: {t/N}')
    					print("_____________________________________________________")


if __name__=="__main__":
	forward_time(32, 10, [4, 6, 8], [2, 4], [64, 128], [8, 16], [1024, 2048])
	