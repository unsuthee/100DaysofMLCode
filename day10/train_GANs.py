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

imageSize = 28
num_classes = 10

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

dataset = 'MNIST'

if dataset == 'FashionMNIST':
    train_set = datasets.FashionMNIST('../datasets/stl10', train=True, download=True, transform=trans)
    nc = 1
elif dataset == 'MNIST':
    train_set = datasets.MNIST('../datasets/mnist', train=True, download=True, transform=trans)
    nc = 1
    
batch_size = 64

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))

os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device("cuda:0")

def weight_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
        
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.z_to_zh = nn.Sequential(nn.Linear(latent_dim, 256),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(inplace=True))
        self.y_to_yh = nn.Sequential(nn.Linear(num_classes, 256),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(inplace=True))
        self.g = nn.Sequential(nn.Linear(512, 512),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 1024),
                               nn.BatchNorm1d(1024),
                               nn.ReLU(inplace=True),
                               nn.Linear(1024, 784),
                               nn.Tanh())
    
    def forward(self, zb, yb):
        zh = self.z_to_zh(zb)
        yh = self.y_to_yh(yb)
        
        z = torch.cat([zh, yh], dim=1)
        output = self.g(z)
        return output.view(-1, 1, 28, 28)
    
    def name(self):
        return "Generator"
    
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
    data = Variable(torch.ones(size))
    return data.cuda()

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size))
    return data.cuda()

def one_hot(label_batch, num_classes):
    yb_onehot = torch.eye(num_classes)[label_batch-1]
    yb_onehot = Variable(yb_onehot)
    return yb_onehot

class Discriminator(nn.Module):
    def __init__(self, num_latent, num_classes):
        super(Discriminator, self).__init__()
        self.num_latent = num_latent
        self.num_classes = num_classes
        
        self.x_to_xh = nn.Sequential(nn.Linear(28*28, 1024),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.y_to_yh = nn.Sequential(nn.Linear(num_classes, 1024),
                                     nn.LeakyReLU(0.2, inplace=True))
        
        self.d = nn.Sequential(nn.Linear(2048, 512),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(512, 256),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(256, 1),
                                  nn.Sigmoid())

    def forward(self, xb, yb):
        xb = xb.view(-1, 28*28)
        xh = self.x_to_xh(xb)
        yh = self.y_to_yh(yb)
        
        xyh = torch.cat([xh, yh], dim=1)
        out = self.d(xyh)
        return out.squeeze(1)
        
    def name(self):
        return "Discriminator"

num_latent = 100
num_classes = 10

D = Discriminator(num_latent, num_classes).to(device)
D.apply(weight_init)

G = Generator(num_latent, num_classes).to(device)
G.apply(weight_init)

D_optimizer = optim.SGD(D.parameters(), lr=0.0001, momentum=0.9)
G_optimizer = optim.Adam(G.parameters(), lr=0.0001)

loss = nn.BCELoss()    

def train_discriminator(optimizer, real_data, fake_data, label_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = D(real_data, label_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = D(fake_data, label_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data, label_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = D(fake_data, label_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

from pathlib import Path
home = str(Path.home())

num_test_samples = 24
test_noise = noise(num_test_samples, nz)

import numpy as np
num_batches = len(train_loader)

# Create logger instance
with open('logs/loss.log', 'w') as log_fn:
    
    log_fn.write('epoch,d_error,g_error,n_batch,num_batches\n')
    
    # Total number of epochs to train
    num_epochs = 500
    for epoch in range(num_epochs):
        for n_batch, (real_batch, label_batch) in enumerate(train_loader):
            N = real_batch.size(0)
            label_data = one_hot(label_batch, num_classes).to(device)
            # 1. Train Discriminator
            real_data = Variable(real_batch)
            real_data = real_data.to(device)

            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            fake_data = G(noise(N, nz), label_data).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                  train_discriminator(D_optimizer, real_data, fake_data, label_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = G(noise(N, nz), label_data)
            # Train G
            g_error = train_generator(G_optimizer, fake_data, label_data)
            # Log batch error
            log_fn.write('{},{:.6f},{:.6f},{},{}\n'.format(epoch, d_error, g_error, n_batch, num_batches))
            
            # Display Progress every few batches
            #if (n_batch) % 100 == 0: 
            #    test_images = G(test_noise)
            #    test_images = test_images.data
                
        print("epoch: {} d_error: {:.4f} g_error: {:.4f}".format(epoch, d_error, g_error))
        if epoch % 5 == 0:
            for n in range(num_classes):
                class_onehot = n * torch.ones(test_noise.size(0)).type(torch.LongTensor)
                class_onehot = torch.eye(num_classes)[class_onehot]
                class_onehot = Variable(class_onehot).cuda()

                test_images = G(test_noise, class_onehot)
                test_images = test_images.data
                np.save('img/generated_img.{}.epoch{}'.format(n, epoch), test_images.cpu().numpy())
                
