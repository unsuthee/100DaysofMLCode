import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

imageSize = 64
batch_size = 64

trans = transforms.Compose([transforms.Resize(imageSize),
                            transforms.CenterCrop(imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = 'CIFAR10'

if dataset == 'CIFAR10':
    train_set = datasets.CIFAR10('../datasets/cifar10', train=True, download=True, transform=trans)
elif dataset == 'STL10':
    train_set = datasets.FashionMNIST('../datasets/stl10', train=True, download=True, transform=trans)


train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))

os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device("cuda:0")
ngpu = 1 # number of gpu to use
nz = 100 # size of the latent z vector int(opt.nz)
ngf = 64
ndf = 64
nc = 3

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
    
    def name(self):
        return "Generator"
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
    def name(self):
        return "Discriminator"

def noise(num_dim, latent_dim):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(num_dim, latent_dim, 1, 1))
    return n.cuda()

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size))
    return data.cuda()

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size))
    return data.cuda()

D = Discriminator(ngpu).to(device)
D.apply(weights_init)

G = Generator(ngpu).to(device)
G.apply(weights_init)

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

num_test_samples = 20
test_noise = noise(num_test_samples, nz)

import numpy as np
num_batches = len(train_loader)

# Create logger instance
with open('logs/loss.log', 'w') as log_fn:
    
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
            fake_data = G(noise(N, nz)).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                  train_discriminator(D_optimizer, real_data, fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = G(noise(N, nz))
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
            np.save('img/generated_img.epoch{}'.format(epoch), test_images.cpu().numpy())
