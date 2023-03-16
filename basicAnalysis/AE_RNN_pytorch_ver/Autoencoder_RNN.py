import scipy.io as sio
import numpy as np
import torch
import torch.utils.data as data_load
from torch.utils.data import random_split
import torch.nn as nn
from LSTM_attn import myLSTM
from sklearn.metrics import f1_score
import os,sys
import os.path as osp
import pandas as pd
os.chdir(sys.path[0])
if not osp.exists('D:/Data/Checkpoints'):
    os.makedirs('D:/Data/Checkpoints')
data = sio.loadmat("D:\\Data\\all.mat")
labels = sio.loadmat("D:\\Data\\all_lab.mat")
data = data['all_com']
labels = labels['all_lab']
alldata = np.empty((np.size(data,1),np.shape(data[0,0])[1],np.shape(data[0,0])[0]))
alllabel = np.empty((np.size(data,1),1))
for i in range(np.size(data,1)):
    alldata[i,:,:] = data[0,i].transpose()
    alllabel[i,] = labels[0,i]-1

alldata = torch.Tensor(alldata)
alllabel = torch.Tensor((np.ravel(alllabel))).long()

F1s = []
Accu_all = []
for ix in range(50):
    print('Now Doing',ix,'Iteration Train-Test')
    torch_dataset = data_load.TensorDataset(alldata,alllabel)
    trainset,valset,testset = random_split(dataset = torch_dataset,lengths= [0.8,0.1,0.1])
    trainloader = data_load.DataLoader(dataset=trainset,batch_size=16,shuffle=True,num_workers=0)
    valloader = data_load.DataLoader(dataset=valset,batch_size=12,shuffle=True,num_workers=0)
    testloader = data_load.DataLoader(dataset=testset,batch_size=12,shuffle=True,num_workers=0)

    print('Finish Loading Data')
    num_epochs = 300
    hidden_size = 256
    input_size = 6
    output_size = 3
    Learningrate = 1e-3

    model = myLSTM(ninput=input_size,nhid=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learningrate)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    model = model.to(device)
    print('All Set, Happy Training!')
    def train():
        model.train()
        bestf1 = 0
        correct = 0
        total = 0
        for epochs in range(num_epochs):
            f1 = 0
            for step, (traindata,trainlabel) in enumerate(trainloader):
                model.train()
                traindata = traindata.to(device)
                trainlabel = trainlabel.to(device)
                optimizer.zero_grad()
                output = model(traindata)
                loss = criterion(output,trainlabel)
                loss.backward()
                optimizer.step()
                _, predict = torch.max(output.data,1)
                correct += (predict==trainlabel).sum().item()
                total += trainlabel.size(0)
                pred = predict.cpu().numpy()
                lab = trainlabel.cpu().numpy()
                f1 += f1_score(pred,lab,average='macro')
                trainaccu = correct/total
                if (step + 1) % 20 == 0:
                    trainf1 = f1/step
                    print("Train Epoch[{}/{}],step[{}/{}],trainaccu{:.3f},train_f1_score:{:.3f},loss:{:.3f}".format(epochs+1,num_epochs,step+1,len(trainloader),trainaccu,trainf1,loss.item()))
                if (step + 1) % 20 ==0:
                    trainf1 = f1/step
                    losseval, evalf1, valaccu = evaluate()
                    if bestf1 < evalf1:
                        bestf1 = evalf1
                        torch.save(model.state_dict(),'D:/Data/Checkpoints/bestmodel.pt')
                    print("Valid Epoch[{}/{}],step[{}/{}],val_accu{:.3f},val_f1_score{:.3f},loss:{:.3f}".format(epochs+1,num_epochs,step+1,len(trainloader),valaccu,evalf1,losseval.item()))

    def evaluate():
        model.eval()
        f1 = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for step,(valdata,vallabel) in enumerate(valloader):
                valdata = valdata.to(device)
                vallabel = vallabel.to(device)
                output = model(valdata)
                loss = criterion(output,vallabel)
                _, predict = torch.max(output.data,1)
                pred = predict.cpu().numpy()
                lab = vallabel.cpu().numpy()
                correct += (predict==vallabel).sum().item()
                total += vallabel.size(0)
                f1 += f1_score(pred,lab,average='macro')
            f1 = f1/step
        return loss,f1,correct/total

    def test():
        best_model = myLSTM(ninput=input_size,nhid=hidden_size)
        best_model.load_state_dict(torch.load('D:/Data/Checkpoints/bestmodel.pt'))
        best_model.to(device)
        best_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            f1 = 0
            for step, (testdata,testlabel) in enumerate(testloader):              
                testdata = testdata.to(device)
                testlabel = testlabel.to(device)
                output = model(testdata)
                loss = criterion(output,testlabel)
                _, predict = torch.max(output.data,1)
                pred = predict.cpu().numpy()
                lab = testlabel.cpu().numpy()
                correct += (predict==testlabel).sum().item()
                total += testlabel.size(0)
                f1 += f1_score(pred,lab,average='macro')

            f1 = f1/step
            return f1, correct/total

    if __name__ == '__main__':
        train()
        testf1, testaccu = test()
        print('Test Accu is',testaccu)
        print('Test F1 is',testf1)
        F1s.append(testf1)
        Accu_all.append(testaccu)

df = pd.DataFrame(data = F1s)
df.to_csv('D:/Data/F1_score.csv',encoding='utf-8')
df = pd.DataFrame(data = Accu_all)
df.to_csv('D:/Data/Accuracy.csv',encoding='utf-8')
print(np.mean(Accu_all))
print(np.mean(F1s))
