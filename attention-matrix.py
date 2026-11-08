import torch.nn as nn
import torch
from classifier import Classifier
from dataset_loader import load_sst2

if __name__=="__main__":
	num_layers=2
	d_e = 64
	d_k = 16
	d_ff = 128
	h = 4
	model = Classifier(num_layers, d_e, d_k, d_k, d_ff, h)
	state_dict = torch.load('data/SST-2/best_model.pt')
	model.load_state_dict(state_dict)
	train_data, val_data = load_sst2(
        max_train_samples=10,
        max_val_samples=10,)
	x = train_data[3][0]
	print(x)
	model.attention_matrix(x)
