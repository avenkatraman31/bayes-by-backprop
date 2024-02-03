# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:00:45 2021

@author: vadit
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from sklearn.datasets import fetch_openml# fetch_mldata
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
device = torch.device("cpu")

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


class MLPLayer(nn.Module):
    '''Linear transformation layer using pre-defined means,variances of weights and biases - 
    Uses Local Reparametrization trick described in Blundell [2015]
    Public access for log prior and log variation used in defining loss function'''
    def __init__(self, n_input, n_output, Variational_Sigma):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.Variational_Sigma = Variational_Sigma
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.W_logsigma = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.b_logsigma = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.lpw = 0
        self.lqw = 0

    def forward(self, X, infer=False):
        if infer:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_output)
            return output

        epsilon_W, epsilon_b = self.get_random()
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
        self.lpw = log_gaussian(W, 0, self.Variational_Sigma).sum() + log_gaussian(b, 0, self.Variational_Sigma).sum()
        self.lqw = log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum()
        return output

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, self.Variational_Sigma).to(device)), Variable(torch.Tensor(self.n_output).normal_(0, self.Variational_Sigma).to(device))


class MLP(nn.Module):
    '''Multi-output Bayesian Neural Network with configurable number of hidden layers, nodes per hidden layer and activation
    in each layer'''
    def __init__(self, n_input, n_hidden,n_output,activation='relu',Variational_Sigma=float(np.exp(-3))):
        import numpy as np
        super().__init__()
        assert len(activation)==len(n_hidden)-1
        for i in range(len(activation)):
            assert activation[i] in ['relu','sigmoid','leakyrelu','relu6']
        self.layers=nn.ModuleList()
        self.n_all=np.append(n_input, n_hidden)
        self.n_all=np.append(self.n_all,n_output)
        print(self.n_all)
        for i in range(len(n_hidden)):
            self.layers.append(MLPLayer(self.n_all[i], self.n_all[i+1], Variational_Sigma))
            if(i==len(n_hidden)-1): 
                self.layers.append(nn.Softmax())
                break
            if(activation[i]=='relu'): self.layers.append(nn.ReLU())
            if(activation[i]=='sigmoid'): self.layers.append(nn.Sigmoid())
            if(activation[i]=='leakyrelu'): self.layers.append(nn.LeakyReLU())
            if(activation[i]=='relu6'): self.layers.append(nn.ReLU6())
        
        # self.l1 = MLPLayer(n_input, 200, Variational_Sigma)
        # self.l1_relu = nn.ReLU()
        # self.l2 = MLPLayer(200, 200, Variational_Sigma)
        # self.l2_relu = nn.ReLU()
        # self.l3 = MLPLayer(200, 10, Variational_Sigma)
        # self.l3_softmax = nn.Softmax()

    def forward(self, X, infer=False):
        output=X
        for i in range(0,len(self.layers),2):
            output=self.layers[i](output,infer)
            output=self.layers[i+1](output)
        # output = self.l1_relu(self.l1(X, infer))
        # output = self.l2_relu(self.l2(output, infer))
        # output = self.l3_softmax(self.l3(output, infer))
        return output

    def get_lpw_lqw(self):
        lpw =sum([self.layers[i].lpw for i in range(0,len(self.layers),2)]) #self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw =sum([self.layers[i].lqw for i in range(0,len(self.layers),2)]) #self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw


def forward_pass_samples(X, y):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in xrange(n_samples):
        output = net(X)
        sample_log_pw, sample_log_qw = net.get_logPriorPDF_logVariationalPDF()
        sample_log_likelihood = log_gaussian(y, output, Variational_Sigma).sum()
        s_log_pw += sample_log_pw
        s_log_qw += sample_log_qw
        s_log_likelihood += sample_log_likelihood

    return s_log_pw/n_samples, s_log_qw/n_samples, s_log_likelihood/n_samples


def criterion(l_pw, l_qw, l_likelihood):
    return ((1./n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)



class VariationalPosteriorLayer(nn.Module):
    '''Linear transformation layer using pre-defined means,variances of weights and biases - 
    Uses Local Reparametrization trick described in Blundell [2015]
    Public access for log prior and log variation used in defining loss function'''
    def __init__(self, n_input, n_output, Variational_Sigma):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.Variational_Sigma = Variational_Sigma
        self.w_mean = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.w_logSTD = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.bias_mean = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.bias_logSTD = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.logPriorPDF = 0
        self.logVariationalPDF = 0

    def forward(self, X, infer=False):
        

        w_rnorm, b_rnorm = self.get_random()
        w = self.w_mean + torch.log(1 + torch.exp(self.w_logSTD))*w_rnorm
        bias = self.bias_mean + torch.log(1 + torch.exp(self.bias_logSTD))*b_rnorm
        output = torch.mm(X, w) + bias.expand(X.size()[0], self.n_output)
        if infer:
            return output
        self.logPriorPDF = log_gaussian(w, 0, self.Variational_Sigma).sum() +\
                           log_gaussian(bias, 0, self.Variational_Sigma).sum()#logPrior of biases
        self.logVariationalPDF = log_gaussian_logsigma(w, self.w_mean, self.w_logSTD).sum() +\
                                 log_gaussian_logsigma(bias, self.bias_mean, self.bias_logSTD).sum()#logVariation of biases
        return output

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, self.Variational_Sigma).to(device)),\
               Variable(torch.Tensor(self.n_output).normal_(0, self.Variational_Sigma).to(device))

class BayesianNetwork(nn.Module):
    '''Multi-output Bayesian Neural Network with configurable number of hidden layers, nodes per hidden layer and activation
    in each layer'''
    def __init__(self, n_input,n_hidden,n_output, activation='relu',Variational_Sigma=float(np.exp(-4))):
        super().__init__()
        if(hasattr(activation,'__len__')): 
            assert len(activation)==len(n_hidden)-1
            self.activation=activation
        else: 
            self.activation=[activation for i in range(len(n_hidden)-1)]
        for act in self.activation:
            assert act in ['relu','sigmoid','leakyrelu','relu6']
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output
        self.Variational_Sigma=Variational_Sigma
        
        self.layers=nn.ModuleList()
        self.n_all=np.append(self.n_input, self.n_hidden)
        self.n_all=np.append(self.n_all,self.n_output)
        print(self.n_all)
        for i in range(len(self.n_hidden)):
            self.layers.append(VariationalPosteriorLayer(self.n_all[i], self.n_all[i+1], self.Variational_Sigma))
            if(i==len(self.n_hidden)-1): 
                self.layers.append(nn.Softmax())
                break
            if(self.activation[i]=='relu'): self.layers.append(nn.ReLU())
            if(self.activation[i]=='sigmoid'): self.layers.append(nn.Sigmoid())
            if(self.activation[i]=='leakyrelu'): self.layers.append(nn.LeakyReLU())
            if(self.activation[i]=='relu6'): self.layers.append(nn.ReLU6())
        

    def forward(self, X, infer=False):
        output=X
        for i in range(0,len(self.layers),2):
            output=self.layers[i](output,infer)
            output=self.layers[i+1](output)
        return output

    def get_logPriorPDF_logVariationalPDF(self):#Must add more if there are more linear layers: must find a better way!
        logPriorPDF = sum([self.layers[i].logPriorPDF for i in range(0,len(self.layers),2)]) 
        logVariationalPDF = sum([self.layers[i].logVariationalPDF for i in range(0,len(self.layers),2)]) 
        return logPriorPDF, logVariationalPDF

mnist = fetch_openml('mnist_784')
N = 5000

data = np.float32(mnist.data[:]) / 255.
idx = np.random.choice(data.shape[0], N)
data = data[idx]
target = np.int32(mnist.target[idx]).reshape(N, 1)

train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.05)
train_data, test_data = data[train_idx], data[test_idx]
train_target, test_target = target[train_idx], target[test_idx]

train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target))

n_input = train_data.shape[1]
M = train_data.shape[0]
Variational_Sigma = float(np.exp(-3))
n_samples = 3
learning_rate = 0.001
n_epochs = 100
n_hidden=np.array([200,1,10],dtype='int')
n_output=1
activation=['relu','relu']
# Initialize network
# net = MLP(n_input, Variational_Sigma).to(device)
# net=MLP(n_input,n_hidden=n_hidden,n_output=n_output,activation=activation,Variational_Sigma=Variational_Sigma).to(device)
net=BayesianNetwork(n_input, n_hidden, n_output,activation=activation,Variational_Sigma=Variational_Sigma).to(device)
# net = net.cuda()

# building the objective
# remember, we're evaluating by samples
log_pw, log_qw, log_likelihood = 0., 0., 0.
batch_size = 100
n_batches = M / float(batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

n_train_batches = int(train_data.shape[0] / float(batch_size))
xrange=range
for e in xrange(n_epochs):
    errs = []
    for b in range(n_train_batches):
        net.zero_grad()
        X = Variable(torch.Tensor(train_data[b * batch_size: (b+1) * batch_size]).to(device))
        y = Variable(torch.Tensor(train_target[b * batch_size: (b+1) * batch_size]).to(device))

        log_pw, log_qw, log_likelihood = forward_pass_samples(X, y)
        loss = criterion(log_pw, log_qw, log_likelihood)
        errs.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

    X = Variable(torch.Tensor(test_data).to(device), volatile=True)
    pred = net(X, infer=True)
    _, out = torch.max(pred, 1)
    acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(test_target.ravel())) / float(test_data.shape[0])

    print ('epoch', e, 'loss', np.mean(errs), 'acc', acc)