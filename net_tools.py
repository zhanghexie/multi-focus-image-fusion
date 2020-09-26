""""
作者：张鹤轩
函数：
    weights_init：权重初始化函数
    get_acc_num：获取判断正确数
    get_trained_net：加载训练好的网络
    train_net：训练网络
    save_net：保存网络
"""
import numpy as np 
import torch 
from data_tools import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2 as cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def weights_init(m):
    """
    权重初始化
    """
    classname = m.__class__.__name__                               
    if classname.find('Conv') != -1:                               
        nn.init.normal_(m.weight.data, 0.0, 0.05)           
    elif classname.find('BatchNorm') != -1:                 
        nn.init.normal_(m.weight.data, 1.0, 0.02)               
        nn.init.constant_(m.bias.data, 0)

def get_acc_num(label,value):
    """
    给出判断正确的数量
    """
    with torch.no_grad():
        return (value==label).sum().item()

def get_trained_net(net,netPath):
    """
    加载训练好的网络
    """  
    
    net = net.to(device)
        
    if device == torch.device("cpu"): 
        net.load_state_dict(torch.load(netPath,map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(netPath))
        
    return net

def train_net(net,trainSet,testSet,epohNum,lossFunction,device,trainLoader=None,testLoader=None,opt=None,hopeAcc=1,minSaveAcc=0.9,printNum=50,path='./save_net/'):
    """
    训练网络
    """
    if trainLoader is None:
        trainLoader = DataLoader(trainSet,batch_size=256,shuffle=True,num_workers=4)
    if testLoader is None:
        testLoader = DataLoader(testSet,batch_size=256,shuffle=True,num_workers=4)
    
    if opt is None:
        opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.apply(weights_init)
    maxAcc = minSaveAcc
    
    for epoh in range(epohNum):
        break_flag = False
        runningLoss = 0.0

        for i, trainData in enumerate(trainLoader, 0):
            
            inputs = trainData[0].to(device)
            labels = trainData[1].float().to(device)

            opt.zero_grad()
            outputs = net(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            opt.step()

            runningLoss += loss.item()
            
            if i % printNum == printNum - 1:
                acc = 0
                for j,testData in enumerate(testLoader, 0):
                    testInput = testData[0].to(device)
                    testLabel = testData[1].to(device)
                    predictValue = net.predict(testInput)
                    acc += get_acc_num(testLabel,predictValue)
                acc = acc/len(testSet)
                if acc >= hopeAcc:
                    temp = path + (str(int(acc*1000))+".pth")
                    save_net(net,temp)
                    break_flag = True
                    break
                else:
                    if acc > maxAcc:
                        temp = path + (str(int(acc*1000))+".pth")
                        save_net(net,temp)
                        maxAcc = acc
                print('[%d, %5d]  loss: %.3f  acc: %.3f' %(epoh+1, i+1, runningLoss/(i+1), acc))
        if break_flag:
            break
    print("train finish !!!")

def save_net(net,path,allNet=False):
    """
    保存网络
    """
    if allNet:
        torch.save(net,path)
    else:
        torch.save(net.state_dict(),path)