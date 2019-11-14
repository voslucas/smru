from scipy.io import loadmat
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
import numpy as np
import smru


class MusicModel(nn.Module):
    def __init__(self, mode, input_size, output_size, hidden_size, num_layers, dropout, bmode, wmode):
        super(MusicModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if mode.startswith("SMRU"):
            self.rnn = smru.SMRU(input_size,hidden_size, num_layers, batch_first=True, mode=mode.upper(), dropout = dropout, bmode=bmode,wmode=wmode)
        elif mode=="GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif mode=="LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            
        else:
            raise Exception('Unknown mode: {}'.format(mode))

        self.linear = nn.Linear(self.hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.rnn(x)
        #
        output = self.linear(output) #.double()
        return self.sig(output)



def data_generator(dataset):
    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat('./mdata/JSB_Chorales.mat')
    elif dataset == "Muse":
        print('loading Muse data...')
        data = loadmat('./mdata/MuseData.mat')
    elif dataset == "Nott":
        print('loading Nott data...')
        data = loadmat('./mdata/Nottingham.mat')
    elif dataset == "Piano":
        print('loading Piano data...')
        data = loadmat('./mdata/Piano_midi.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test


parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--rnn_type', type=str, default="SMRU5",
                    help='RNN Cell : LSTM, GRU, SMRU1, .. SMRU4 ')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip, -1 means no clip (default: 0.8)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--layers', type=int, default=2,
                    help='# of layers (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='Nott',
                    help='the dataset to run (default: Nott)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1111)')
parser.add_argument('--wmode', type=str, default="xn",
                    help='smru weight initialization: xn,xu,id,nn ')
parser.add_argument('--bmode', type=str, default="bz",
                    help='smru bias initialization: bk,ba,bf,bz')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
input_size = 88
X_train, X_valid, X_test = data_generator(args.data)

hidden_size = args.nhid
num_layers = args.layers
dropout = args.dropout
mode = args.rnn_type
task = "nott"
bmode = args.bmode
wmode = args.wmode

model = MusicModel(mode,input_size, input_size, hidden_size, num_layers, dropout=args.dropout, bmode=bmode,wmode=wmode)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x.unsqueeze(0)).squeeze(0)
            loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                                torch.matmul((1-y), torch.log(1-output).float().t()))
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        #print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss


def train(ep):
    model.train()

    total_loss = 0
    count = 0

    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])

        optimizer.zero_grad()
        tmp = x.unsqueeze(0)
        output = model(x.unsqueeze(0)).squeeze(0)

        loss = -torch.trace(torch.matmul(y, torch.log(output).t()) +
                            torch.matmul((1 - y), torch.log(1 - output).t()))

        # SKIP - NO INBETWEEN EPOCH RESULTS NEEDED FOR TASK-RUNNER AUTOMATED VERSION OF THIS TES
        # total_loss += loss.item()
        # count += output.size(0)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()

        # SKIP - NO INBETWEEN EPOCH RESULTS NEEDED FOR TASK-RUNNER AUTOMATED VERSION OF THIS TEST 
        #if idx > 0 and idx % args.log_interval == 0:
        #    cur_loss = total_loss / count
        #    print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
        #    total_loss = 0.0
        #    count = 0


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    model_name = "poly_music_{0}.pt".format(args.data)
    for ep in range(1, args.epochs+1):
        train(ep)
        vloss = evaluate(X_valid, name='Validation')
        tloss = evaluate(X_test, name='Test')
        if vloss < best_vloss:
            # SKIP - NOT NEEDED
            # with open(model_name, "wb") as f:
            #     torch.save(model, f)
            #     print("Saved model!\n")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)
        print("{};{};{};{};{};{};{};{};{:.5f};{:.5f};{};{};".format(task,mode,lr, hidden_size,num_layers, args.clip ,args.seed,ep, vloss, tloss, bmode, wmode))


    # SKIP - NO FINAL CONCLUSION NEEDED IN OUR OUTPUT
    # print('-' * 89)
    # model = torch.load(open(model_name, "rb"))
    # tloss = evaluate(X_test)


