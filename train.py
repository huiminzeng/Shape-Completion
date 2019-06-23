from encoder_naive import *
from encoder_unet import *
from dataloader_new import *
from solver import Solver

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

EPOCH = 50
BATCH_SIZE = 16
LOG_NTH = 100

# load all training data
training_data = MyDataset(flag = 'train')
train_loader = torch.utils.data.DataLoader(training_data, batch_size=16, shuffle=False)

validataion_data = MyDataset(flag = 'val')
val_loader = torch.utils.data.DataLoader(validataion_data, shuffle=False, batch_size=BATCH_SIZE)


test_data = MyDataset( flag = 'test')
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

solver = Solver()
model = generator()

solver.train(model, train_loader, val_loader, num_epochs=EPOCH, log_nth=LOG_NTH)
#print('file: ', file_num)
torch.save(model, 'models/model')