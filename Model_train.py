#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""

import os
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSE, Score



# ==================== 1: Parameters for training ============================
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(42)
batch_size = 16
epochs = 1000
learning_rate = 3e-4
num_workers = 4
print_freq = 100  # print frequency (default: 60)
# parameters for data
feedback_bits = 512
all_data = False


# ==================== 2: Model construction ============================
model = AutoEncoder(feedback_bits)
if use_single_gpu:
    model = model.cuda()

else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-9, last_epoch=-1)


# ==================== 3: load dataset ============================
data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']  # shape=8000*126*128*2

x_train = np.transpose(x_train.astype('float32'),[0,3,1,2])
print(np.shape(x_train))
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']  # shape=2000*126*128*2

x_test = np.transpose(x_test.astype('float32'),[0,3,1,2])
print(np.shape(x_test))
# Data loading



# dataLoader for training
if all_data==True:
    train_dataset = DatasetFolder(np.concatenate([x_train, x_test]))
else:
    train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)



# ==================== 4: model training ============================
best_loss = 1
for epoch in range(epochs):
    # model training
    model.train()
    for i, input in enumerate(train_loader):
        # adjust learning rate
        if epoch == 300:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * 0.1
        input = input.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, input)
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.6f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()))
    # model evaluating
    model.eval()
    total_loss = 0
    nmse = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            total_loss += criterion(output, input).item() * input.size(0)
            if epoch % 10 == 0:
                nmse += NMSE(np.transpose(input.cpu().numpy(), (0, 2, 3, 1)), np.transpose(output.cpu().numpy(), (0, 2, 3, 1)))
        if epoch % 10 == 0:
            print('Score {}'.format(Score(nmse / len(test_loader))))
        average_loss = total_loss / len(test_dataset)
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder.pth.tar'
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 = './Modelsave/decoder.pth.tar'
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            print("Model saved")
            best_loss = average_loss
scheduler.step()
