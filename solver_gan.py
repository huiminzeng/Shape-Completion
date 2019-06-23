import numpy as np 
import torch
import time
import os
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from tensorboardX import SummaryWriter

class Solver(object):
    default_adam_args ={ 'lr': 1e-4,
                         'betas': (0.9, 0.999),
                         'eps': 1e-8,
                         'weight_decay': 0.0
    }
    def __init__(self, optim=torch.optim.Adam, optim_args={}, 
                loss_func=torch.nn.L1Loss(reduction='sum')):

        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)

        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

        self.writer = SummaryWriter('/home/huiminzeng/Desktop/Code/Recovery/TensorBoardX')

        self.best_model = None
        self.best_valloss = 10000000
        self.lr = self.optim_args['lr']

    
    def _reset_histories(self):
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_loss_plot = []
        self.val_loss_plot = []
        self.model = []
        
    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        model.cuda()

        print("TRAINING START!!!!!!!!!!")
        print()

        for epoch in range(num_epochs):
            
            if epoch < 7:
                lr = self.lr
                optim = self.optim(model.parameters(), lr)
            elif epoch < 15:
                lr = self.lr * 0.5
                optim = self.optim(model.parameters(), lr) 
            else:
                lr = self.lr * 0.25
                optim = self.optim(model.parameters(), lr)
            print('learning rate: ', lr)
            

            #TRAINING
            epo_start = time.time()
            model.train()
            for i, (x_train,y_train) in enumerate(train_loader):

                #if overfitting
                if i > 0:
                    break
                model.train()
                optim.zero_grad()

                iter_start = time.time()

                if torch.cuda.is_available():
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()

                mask = x_train[:, [-1]].eq(-1).float()
                output_train = model(x_train)

                output_masked = output_train*mask
                y_masked = y_train*mask
            
                loss = self.loss_func(output_masked, y_masked) 
                #print('loss: ', loss.data.cpu().numpy())
                forward_end = time.time()

                #print("forward pass:", forward_end - iter_start)
                loss.backward()
                backward_end = time.time()
                optim.step()
                #print("back propagation: ", backward_end-forward_end)
                loss_value = loss.data.cpu().numpy()/(len(train_loader))

                self.writer.add_scalar('train_loss/iteration', loss_value, i+epoch*iter_per_epoch)


                self.train_loss_history.append(loss_value)
                
                
                if log_nth and i % log_nth == 0 and i!=0:
                    print(i)
                    print()
                    if i == 0:
                        print("TRAINING VISUALIZATION!!!!!!!!!!")
                        v_time1 = time.time()
                        voxels_train = self.get_voxel(x_train, y_train, output_train)
                        self.voxel_plot(voxels_train[0], voxels_train[1], voxels_train[2])
                        v_time2 = time.time()
                        print("time: ", v_time2-v_time1)
                    
                    print()
                    print('ITERATION TIME CONSUMPTION! ', backward_end-iter_start)

                    last_log_nth_losses = np.mean(self.train_loss_history[-log_nth:])

                    self.writer.add_scalar('train_log/iteration', last_log_nth_losses, i+epoch*iter_per_epoch)

                    self.train_loss_plot.append(last_log_nth_losses)
                    train_loss = last_log_nth_losses
                    print('logged: [Iteration %d/%d] TRAIN LOSS: %.3f'  
                        % ((i + iter_per_epoch * epoch) , (num_epochs * iter_per_epoch), train_loss))

                
            self.validation(model, val_loader, i, epoch, iter_per_epoch)

            epo_end = time.time()
            epo_time = epo_end - epo_start
            print('[EPOCH %d] time: %.3f' % (epoch, epo_time))

    def validation(self, model, val_loader, iteration, epoch, iter_per_epoch):
        model.eval()
        val_time1 = time.time()

        for step, (x_val,y_val) in enumerate(val_loader):

        
            x_val = x_val.cuda()
            y_val = y_val.cuda()

            mask = x_val[:, [-1]].eq(-1).float()

            output_val = model(x_val)

            #close1 = output_val[:,[0]] <= 2
            #close2 = y_val[:,[0]] <= 2

            #mask = (unknown & (close1 | close2)).float()

            output_masked = output_val*mask
            y_masked = y_val*mask
            
            loss = self.loss_func(output_masked, y_masked)/6250      
            self.val_loss_history.append(loss.data.cpu().numpy())



        val_time2 = time.time()
        print("VAL TIME: ", val_time2 - val_time1)
        val_loss = np.mean(self.val_loss_history[-len(val_loader):])

        self.writer.add_scalar('val_log/iteration', val_loss, iteration + iter_per_epoch * epoch)

        if val_loss < self.best_valloss:
            self.best_model = model
            self.best_valloss = val_loss
            torch.save(self.best_model, 'models/best_model')

        self.val_loss_plot.append(val_loss)
        print('logged: [Iteration %d/Epoch %d] VAL LOSS: %.3f' % ((iteration + iter_per_epoch * epoch), epoch, self.val_loss_plot[-1])) 

        
        print("VALIDATION VISUALIZATION!!!!!!!!!!")
        voxels_val = self.get_voxel(x_val, y_val, output_val)
        self.voxel_plot(voxels_val[0], voxels_val[1], voxels_val[2])
        

    def get_voxel(self,x ,y , output):
        voxel_input = x[3][0].cpu().detach().numpy()
        #print(voxel_input.shape)
        voxel_target = y[3].cpu().detach().numpy().squeeze()
        #print(voxel_target.shape)
        voxel_output = output[3].cpu().detach().numpy().squeeze()
        #print(voxel_output.shape)
        return voxel_input, voxel_target, voxel_output
    
    def voxel_plot(self, voxel_input, voxel_target, voxel_output):
        x, y, z = np.indices((32, 32, 32))

        vinput = voxel_input < 0.7
        vtarget = voxel_target < 0.7
        voutput = voxel_output < 0.7


        fig = plt.figure(figsize=plt.figaspect(0.3))

        #===============
        #  First subplot
        #===============
        # set up the axes for the first plot
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        colors = np.empty(voxel_input.shape, dtype=object)
        plt.title("input")
        ax.voxels(vinput, facecolors=colors, edgecolor='k')

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        plt.title("prediction")
        ax = fig.gca(projection='3d')
        ax.voxels(voutput, facecolors=colors, edgecolor='k')

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        plt.title("target")
        ax = fig.gca(projection='3d')
        ax.voxels(vtarget, facecolors=colors, edgecolor='k')
        
        plt.show()

    def test(self, test_loader):
        test_prediction = []
        test_loss_history = []
        model = self.best_model
        model.eval()
        model.cuda()
        for i, (x_test,y_test) in enumerate(test_loader, 1):
            if torch.cuda.is_available():
                x_test = x_test.cuda()
                y_test = y_test.cuda()

            output_test = model(x_test)
            test_prediction.append(output_test.cpu().detach().numpy())
            #print(test_prediction[i-1].shape)
                    
            loss = self.loss_func(output_test, y_test) 
            test_loss_history.append(loss.data.cpu().numpy())

        overall_test_loss = np.mean(test_loss_history)
        return overall_test_loss, test_prediction


