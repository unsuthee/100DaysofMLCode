import os, sys
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

###############################################################################################

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../datasets/mnist', train=True, download=True, transform=trans)
test_set = datasets.MNIST('../datasets/mnist', train=False, download=True, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))

os.environ["CUDA_VISIBLE_DEVICES"]="1"

###############################################################################################

class NADE(nn.Module):
    
    def __init__(self, num_feas, num_hidden_dim):
        super(NADE, self).__init__()
        
        self.num_feas = num_feas
        self.num_hidden_dim = num_hidden_dim

        self.C = Parameter(torch.randn(1, num_hidden_dim))
        self.W = Parameter(torch.randn(num_feas, num_hidden_dim))
        self.B = Parameter(torch.randn(num_feas))
        
    def forward(self, batch_x):
        loss = []
        for x in batch_x:
            binary_x = Variable((x > 0).type(torch.cuda.FloatTensor))
            prob_mat = Variable(torch.empty(self.num_feas).type(torch.cuda.FloatTensor))
            prob_mat[0] = F.sigmoid(torch.mv(self.C, self.W[0]) + self.B[0])
            for i in range(1, self.num_feas):
                t = binary_x[:i].unsqueeze(0)
                h = F.sigmoid(torch.mm(t, self.W[:i]) + self.C)
                prob_mat[i] = F.sigmoid(torch.mv(h, self.W[i]) + self.B[i])
            loss.append(F.binary_cross_entropy(prob_mat, binary_x, size_average=False))
        total_loss = torch.stack(loss, dim=0).mean()
        return total_loss
    
    def sample(self):
        prob = F.sigmoid(torch.mv(self.C, self.W[0]) + self.B[0])
        x_ = torch.empty(self.num_feas).type(torch.cuda.FloatTensor)
        x_[0] = torch.bernoulli(prob)
        for i in range(1, self.num_feas):
            t = x_[:i].unsqueeze(0)
            h = F.sigmoid(torch.mm(t, self.W[:i]) + self.C)
            prob = F.sigmoid(torch.mv(h, self.W[i]) + self.B[i])
            x_[i] = torch.bernoulli(prob)
        return x_
        
###############################################################################################

model = NADE(784, 128)
model = model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    i = 0
    for batch_x, batch_y in tqdm(train_loader):
        batch_x = batch_x.view(-1, 784)

        optimizer.zero_grad()

        loss = model(batch_x.cuda())
        loss.backward()

        optimizer.step()

        #torch.save(model.state_dict(), 'NADE.ip.model') 

        i+=1
        if i % 50 == 0:
            print("{}:{}".format(epoch, loss.item()))
            torch.save(model.state_dict(), 'save_models/NADE.{}.{}.model'.format(epoch, i)) 

