from encoder_naive import *
from encoder_unet import *
from dataloader_new import *
from solver import Solver

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

    plt.savefig('GAN_OUTPUT/ %.d' %epoch)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--log_nth", type=int, default=10, help="intervals between outputs")
opt = parser.parse_args()
print(opt)
print(opt.n_epochs)

writer = SummaryWriter('/home/huiminzeng/Desktop/Code/Recovery/GAN_LOSS')

cuda = True if torch.cuda.is_available() else False

# load all training data
training_data = MyDataset(flag = 'train')
train_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size, shuffle=False)

validataion_data = MyDataset(flag = 'val')
val_loader = torch.utils.data.DataLoader(validataion_data, shuffle=False, batch_size=opt.batch_size)


test_data = MyDataset( flag = 'test')
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=opt.batch_size)

# Loss functions
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# Initialize generator and discriminator
generator = Generator().cuda()
discriminator = Discriminator().cuda()


# Optimizers
G_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
D_losses = []
G_losses = []

print('training start!')
start_time = time.time()
# ----------
#  Training
# ----------

total_step = len(train_loader)
print("total_step: ", total_step)

PATH = "GAN_MODEL/"

for epoch in range(opt.n_epochs):
    """
    if epoch < 10:
        pass
    elif epoch < 20 and epoch >= 10:
        G_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr * 0.1, betas=(opt.b1, opt.b2))
        D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr * 0.1, betas=(opt.b1, opt.b2))
    elif epoch < 30 and epoch >= 20:
        G_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr * 0.01, betas=(opt.b1, opt.b2))
        D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr * 0.01, betas=(opt.b1, opt.b2))
    """
    epoch_start_time = time.time()
    num_iter = 0

    for i, (inputs, targets) in enumerate(train_loader):
        iter_start_time = time.time()
        #print("mmp!", inputs.shape)

        batch_size = inputs.shape[0]

        # Adversarial ground truths
        #valid = Variable(torch.ones(batch_size)).cuda()
        #fake = Variable(torch.zeros(batch_size)).cuda()
        valid = torch.Tensor(batch_size).uniform_(0.7, 1.2).cuda()
        fake = torch.Tensor(batch_size).uniform_(0, 0.3).cuda()

        #######################################################
        ##############TRAING DISCRIMINATOR#####################
        #######################################################
        inputs = inputs.cuda()
        targets = targets.cuda()

        discriminator.zero_grad()

        #real loss for discriminator
        D_real_results = discriminator(inputs, targets)
        D_real_loss = BCE_loss(D_real_results, valid)

        #fake loss for discriminator
        G_results = generator(inputs)
         
        D_fake_results = discriminator(inputs, G_results)
        D_fake_loss = BCE_loss(D_fake_results, fake)

        #train loss for discriminator
        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss.backward()
        D_optimizer.step()

        D_train_loss_value = D_train_loss.data.cpu().detach().item()
        writer.add_scalar('D_loss/epoch', D_train_loss_value, i + epoch*total_step)

        #print("loss value: ", D_train_loss_value)
        train_hist['D_losses'].append(D_train_loss_value)

        D_losses.append(D_train_loss_value)

        #######################################################
        ################TRAIN GENERATOR########################
        #######################################################
        generator.zero_grad()

        G_results = generator(inputs)
        D_fake_results = discriminator(inputs, G_results)

        G_train_loss = BCE_loss(D_fake_results, valid)

        G_train_loss.backward()
        G_optimizer.step()

        G_train_loss_value = G_train_loss.data.cpu().detach().item()
       # if G_train_loss_value == 0:
            #print(G_results)
        writer.add_scalar('G_loss/epoch', G_train_loss_value, i + epoch*total_step)

        train_hist['G_losses'].append(G_train_loss_value)

        G_losses.append(G_train_loss_value)

        print('Train_Epoch [{}/{}], Step [{}/{}], D_Train_Loss: {:.4f}, G_Train_Loss: {:.4f}'
                .format(epoch, opt.n_epochs, i + epoch*total_step, total_step*opt.n_epochs, D_train_loss_value, G_train_loss_value))

        iter_end_time = time.time()
        print("iteration time: ", iter_end_time - iter_start_time)


    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print("epoch time: ", per_epoch_ptime)
    torch.save(generator, PATH + "generator" +str(epoch))
    torch.save(discriminator, PATH + "discriminator" + str(epoch))

    plot(inputs.cpu().detach().data.numpy()[0], G_results.cpu().detach().data.numpy()[0], targets.cpu().detach().data.numpy()[0], epoch)

print("done!")
