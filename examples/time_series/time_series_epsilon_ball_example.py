'''
Epsilon ball example using geometric certificates and a simple univariate time series example.
'''

# Brute force copy-paste from "2d_example.py"
import sys 
sys.path.append('..')
sys.path.append('../mister_ed')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn
import matplotlib.patches as patches
from scipy.spatial import HalfspaceIntersection

seaborn.set(font_scale=2)
seaborn.set_style("white")

import numpy as np 
# %matplotlib inline
seaborn.set(font_scale=2)
seaborn.set_style("white")
from geocert import GeoCert
from plnn import PLNN

from pyts import datasets

# Training block -- train a simple 2d Network 
# --- define network 
plnn_obj = PLNN(layer_sizes=[60, 100, 50, 20, 6])

net = plnn_obj.net
# net = torch.nn.Sequential(plnn_obj.net, torch.nn.Softmax())

# Loading dataset
dataset_name = 'SyntheticControl'

data_train, data_test, target_train, target_test = datasets.fetch_ucr_dataset(dataset_name, return_X_y=True)

X = torch.Tensor(np.array(data_train))

y = torch.Tensor(np.array(target_train-1)).long()
# y = torch.nn.functional.one_hot(y-1)

# Training the default network
opt = optim.Adam(net.parameters(), lr=1e-3)

for i in range(3000):
    out = net(Variable(X))
    out
    loss = nn.CrossEntropyLoss()(out, Variable(y))
    
    error = (out.max(1)[1].data != y).float().mean()
    if i % 100 == 0: 
        print(loss.item(), error)
    
    opt.zero_grad()
    (loss).backward()
    opt.step()