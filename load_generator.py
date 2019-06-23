from dataloader_new import *

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import argparse
import os
import math

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def isosurface(M, v, step):
    """
    returns vertices and faces from the isosurface of value v of M,
    subsetting M with the steps argument
    """
    from skimage import measure
    
    M = M.squeeze(0)
    #print(M.shape)

    sel = np.arange(0, np.shape(M)[0], step)
    m = M[np.ix_(sel, sel, sel)]
    verts, faces, _, _ = measure.marching_cubes_lewiner(m, v,
                                                        spacing=(1.0, 1.0, 1.0))

    return verts, faces



def plot(inputs, results, targets, epoch):
    fig = plt.figure("overfitting", figsize=(24, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Input', y=1.1)
    verts1, faces1 = isosurface(inputs, 1, 1)
    coll1 = Poly3DCollection(verts1[faces1],  linewidths=0.1, edgecolors='k')
    ax1.add_collection(coll1)
    ax1.view_init(elev=150, azim=-120)
    ax1.set_xlim(0,32)
    ax1.set_ylim(0,32)
    ax1.set_zlim(0,32)

    ax1 = fig.add_subplot(132, projection='3d')
    ax1.set_title('targets', y=1.1)
    verts2, faces2 = isosurface(targets, 1, 1)
    coll2 = Poly3DCollection(verts2[faces2],  linewidths=0.1, edgecolors='k')
    ax1.add_collection(coll2)
    ax1.view_init(elev=150, azim=-120)
    ax1.set_xlim(0,32)
    ax1.set_ylim(0,32)
    ax1.set_zlim(0,32)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Prediction', y=1.1)
    verts3, faces3 = isosurface(results, 1, 1)
    coll3 = Poly3DCollection(verts3[faces3], linewidths=0.1, edgecolors='k')
    ax3.add_collection(coll3)
    ax3.view_init(elev=150, azim=-120)
    ax3.set_xlim(0,32)
    ax3.set_ylim(0,32)
    ax3.set_zlim(0,32)

    plt.savefig('GAN_FINAL_OUTPUT/ %.d' %epoch)

 

writer = SummaryWriter('/home/huiminzeng/Desktop/Code/Recovery/GAN_LOSS')

cuda = True if torch.cuda.is_available() else False

# load all training data
validataion_data = MyDataset(flag = 'val')
val_loader = torch.utils.data.DataLoader(validataion_data, shuffle=False, batch_size=12)

generator = torch.load("/home/huiminzeng/Desktop/Code/Recovery/GAN_MODEL/generator1")
generator.eval()

print('generating objects!')
start_time = time.time()
# ----------
#  Training
# ----------

for i, (inputs, targets) in enumerate(val_loader):
    iter_start_time = time.time()

    batch_size = inputs.shape[0]

    inputs = inputs.cuda()
    targets = targets.cuda()

    G_results = generator(inputs)

    plot(inputs.cpu().detach().data.numpy()[0], G_results.cpu().detach().data.numpy()[0], targets.cpu().detach().data.numpy()[0], i)

    if i == 10:
        break
print("done!")
