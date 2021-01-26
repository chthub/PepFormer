import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
import math

def genData(file,max_len):
    aa_dict={'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,
             'L':11,'K':12,'M':13,'F':14,'P':15,'O':16,'S':17,'U':18,'T':19,
             'W':20,'Y':21,'V':22,'X':23}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
        
    long_pep_counter=0
    pep_codes=[]
    labels=[]
    for pep in lines:
        pep,label=pep.split(",")
        labels.append(int(label))
        if not len(pep) > max_len:
            current_pep=[]
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
    print("length > 81:",long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes,batch_first=True)
    return data,torch.tensor(labels)

data,label=genData("./dataset/Homo_sapiens.csv",81)
print(data.shape,label.shape)

train_data,train_label=data[:70000],label[:70000]
test_data,test_label=data[70000:],label[70000:]

train_dataset = Data.TensorDataset(train_data, train_label)
test_dataset = Data.TensorDataset(test_data, test_label)
batch_size=256
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class newModel(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.hidden_dim = 25
        self.batch_size = 256
        self.emb_dim = 512
        
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.emb_dim, dropout=0.25)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=3, 
                               bidirectional=True, dropout=0.2)
        
        # 4200,3450
        self.fn=nn.Linear(4100,2)
    
    def forward(self, x):
        x=self.embedding(x)*math.sqrt(self.emb_dim)
        x=self.pos_encoder(x)
        output=self.transformer_encoder(x).permute(1, 0, 2)
        output,hn=self.gru(output)
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        output=output.reshape(output.shape[0],-1)
        hn=hn.reshape(output.shape[0],-1)
        output=torch.cat([output,hn],1)
#         print(output.shape,hn.shape)
        return self.fn(output)



    
device = torch.device("cuda",1)

net=newModel().to(device)
lr = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
loss = nn.CrossEntropyLoss()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x,y=x.to(device),y.to(device)
        outputs=net(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def to_log(log):
    with open("./modelLog.log","a+") as f:
        f.write(log+'\n')
        
epochs=1000
t0=time.time()
for epoch in range(1,epochs+1):
    loss_ls=[]
    net.train()
    for x,y in train_iter:
        x,y=x.to(device),y.to(device)
        output=net(x)
        l = loss(output,y)
        optimizer.zero_grad() 
        l.backward()
        optimizer.step()
        loss_ls.append(l.item())
    
    net.eval() 
    with torch.no_grad(): 
        train_acc=evaluate_accuracy(train_iter,net)
        test_acc=evaluate_accuracy(test_iter,net)
    run_log=f'epoch {epoch}, loss: {np.mean(loss_ls):.4f}, train_acc: {train_acc:.3f}, test_acc: {test_acc:.3f} time: {time.time()-t0:.2f}'
    print(run_log)
    to_log(run_log)