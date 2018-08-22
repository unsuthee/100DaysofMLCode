import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../datasets/mnist', train=True, download=True, transform=trans)
test_set = datasets.MNIST('../datasets/mnist', train=False, download=True, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
 
    def forward(self, input):
        return input.view(*self.shape)
    
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim

        self.g = nn.Sequential(nn.Linear(self.latent_dim, 128*7*7),
                    Reshape(-1,128,7,7),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(inplace=True),
                    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                    nn.Tanh())
        
    def forward(self, z):
        return self.g(z)
    
    def name(self):
        return "Generator"

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
    
        self.latent_dim = latent_dim
                
        self.d = nn.Sequential(nn.Conv2d(1, latent_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(inplace=True),
                    Reshape(-1, 128*7*7),
                    nn.Linear(128*7*7, 128),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(128, 1),
                    nn.Sigmoid())
        
        
    def forward(self, x):
        return self.d(x)
    
    def name(self):
        return "Discriminator"

def noise(num_dim, latent_dim):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(num_dim, latent_dim))
    return n.cuda()

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data.cuda()

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data.cuda()

os.environ["CUDA_VISIBLE_DEVICES"]="1"

latent_dim = 64

D = Discriminator(latent_dim)
D = D.cuda()

G = Generator(latent_dim)
G = G.cuda()

D_optimizer = optim.SGD(D.parameters(), lr=0.0001, momentum=0.9)
G_optimizer = optim.Adam(G.parameters(), lr=0.0001)

loss = nn.BCELoss()

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = D(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = D(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = D(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

from pathlib import Path
home = str(Path.home())

num_test_samples = 16
test_noise = noise(num_test_samples, latent_dim)

import numpy as np
num_batches = len(train_loader)

# Create logger instance
with open('gans_log/loss.log', 'w') as log_fn:
    
    log_fn.write('epoch,d_error,g_error,n_batch,num_batches\n')
    
    # Total number of epochs to train
    num_epochs = 500
    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(train_loader):
            N = real_batch.size(0)
            # 1. Train Discriminator
            real_data = Variable(real_batch)
            real_data = real_data.cuda()

            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            fake_data = G(noise(N, latent_dim)).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                  train_discriminator(D_optimizer, real_data, fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = G(noise(N, latent_dim))
            # Train G
            g_error = train_generator(G_optimizer, fake_data)
            # Log batch error
            log_fn.write('{},{:.6f},{:.6f},{},{}\n'.format(epoch, d_error, g_error, n_batch, num_batches))
            
            # Display Progress every few batches
            if (n_batch) % 100 == 0: 
                test_images = G(test_noise)
                test_images = test_images.data
                
        print("epoch: {} d_error: {:.4f} g_error: {:.4f}".format(epoch, d_error, g_error))
        if epoch % 5 == 0:
            np.save('gans_log/generated_img.epoch{}'.format(epoch), test_images.cpu().numpy())

    