import torch
import torch.nn as nn

class myLSTM(nn.Module):
    def __init__(self,ninput,nhid):
        super(myLSTM,self).__init__()
        self.LSTM = nn.LSTM(input_size=ninput,hidden_size=nhid,num_layers=1,bidirectional=True,dropout=0)
        self.GRU = nn.GRU(input_size=ninput,hidden_size=nhid,num_layers=1,bidirectional=True)
        self.dense = nn.Linear(2*nhid,2*nhid)
        self.dropout = nn.Dropout(0.1)
        self.attn = Attention()
        self.Linear = nn.Linear(2*nhid,3)
        self.classfication = nn.Softmax()
    
    def forward(self,input_data):
        output, lstmhid = self.LSTM(input_data)
        gruout, gruhid = self.GRU(input_data)
        #gruout = self.dense(output)
        output, attn1 = self.attn(output,gruout,output,scale=1)
        self.attn_weight = attn1
        output = self.dropout(output)
        linear_out = self.Linear(output)
        linear_output = torch.squeeze(torch.mean(linear_out,dim=1),dim=1)
        return linear_output

class Attention(nn.Module):
    def __init__(self,attention_dropout=0.1):
        super(Attention,self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,q,k,v,scale,attn_mask=None):
        attention = torch.matmul(q,k.transpose(-1,-2))
        attention = attention*scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask,1e-9)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        context = torch.matmul(attention,v)
        return context, attention