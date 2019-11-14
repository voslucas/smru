import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import smru 
import math


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--modelname', type=str, required=True)
parser.add_argument('--rate', type=float, required=True)
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--layers', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--clipping', type=float, required=True)
parser.add_argument('--sequence' , type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--bmode', type=str, required=True)  
parser.add_argument('--wmode', type=str, required=True)  

debug = False
if debug:
    args = lambda: None
    args.modelname = "smru5"
    args.sequence = 56
    args.seed = 3
    args.rate = 0.01
    args.hidden = 8
    args.layers= 1
    args.clipping = 1.0
    args.epochs = 10
    args.bmode = "bk"
    args.wmode = "xn"
else:
    args = parser.parse_args()

seed = args.seed           #1..10
torch.manual_seed(seed)


cuda = True if torch.cuda.is_available() else False

# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    
Tensor = torch.FloatTensor

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
 
test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())
 
batch_size = 100
num_epochs = args.epochs
n_iters =  num_epochs * (len(train_dataset) / batch_size)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


device = 'cpu'

class SMRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, mode, bmode,wmode):
        super(SMRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.smru = smru.SMRU(input_size,hidden_size,num_layers, batch_first=True, mode=mode , bmode=bmode,wmode=wmode)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=False).to(device) 

        out, _ = self.smru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=False).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
     
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out    

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=False).to(device) 
        
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

wmode = args.wmode
bmode = args.bmode
hidden_dim = args.hidden   #128
layer_dim = args.layers     #1
output_dim = 10
seq_dim = args.sequence  #112 #112 #784 #112 # 28 
input_dim = int(784 / seq_dim)

clipping = args.clipping   #0 0.5 1.0
modelname = args.modelname #smru, gru, lstm

task = "mnist" + str(seq_dim)


if modelname.startswith("smru"):
    model = SMRUModel(input_dim, hidden_dim, layer_dim, output_dim, mode=modelname.upper(), bmode=bmode, wmode=wmode)
elif modelname=="gru":
    model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
elif modelname=="lstm":
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
else:
  raise Exception('Unknown mode: {}'.format(modelname))


criterion = nn.CrossEntropyLoss()
learning_rate = args.rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


iter = 0
for epoch in range(num_epochs):

    start = datetime.datetime.now()
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
      
        images = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)
          
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Clip if necessary
        if clipping>0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

        # Updating parameters
        optimizer.step()
        
        #experiment.log_metric("loss", loss.item(),iter)
        
        iter += 1
        #print('Iteration: {}. Loss: {}. '.format(iter, loss.item()))
         
    
    stop = datetime.datetime.now()
    diff = stop-start
    duration = int(diff.total_seconds() * 1000)

    with torch.no_grad():
        # Calculate Accuracy         
        correct = 0
        total = 0
        # Iterate through test dataset
        for j, (images, labels) in enumerate(test_loader):
            images = Variable(images.view(-1 , seq_dim, input_dim))
                
            # Forward pass only to get logits/output
            outputs = model(images)
                
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
                 
            # Total number of labels
            total += labels.size(0)
            correct += (predicted == labels).sum()
             
        accuracy = 100.0 * correct / total
        #experiment.log_metric("acc", accuracy ,iter)
        # Print Loss
        # print('Epoch: {}. Loss: {}. Accuracy: {}. Wrong :{}'.format(epoch, loss.item(), accuracy, total-correct))
        print("{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(task,modelname,learning_rate, hidden_dim,layer_dim,clipping,seed,epoch+1, loss.item(), accuracy, total-correct,duration,bmode,wmode))
        
