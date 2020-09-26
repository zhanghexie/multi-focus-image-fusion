""""
作者：张鹤轩
类：
    FusionData：数据集类
    ClassficationNet：网络类
"""
import numpy as np 
import torch 
from img_tools import *
from data_tools import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2 as cv
 
 
class FusionData(Dataset):
    """
    数据集类
    """
    def __init__(self, data_dir,istrain, transform=None):
        """
        初始化数据集
        参数：
            root_dir：数据集路径
            transform：要对数据进行的变换
        """
        
        if istrain:
            self.image_dir = data_dir + "train_set/"
            self.data_dir = data_dir + "train_set.pickle"
        else :
            self.image_dir = data_dir + "test_set/"
            self.data_dir = data_dir + "test_set.pickle"
            
        self.transform = transform
        self.data = unpickle(self.data_dir)
 
    def __len__(self):
        """
        返回数据长度
        """
        return len(self.data)
 
    def __getitem__(self, idx):
        """
        实现按索引返回数据和标签的功能
        参数：
            idx：索引
        返回值：
            data：处理好的图片数据
            label：图片的标签
        """
        path = self.image_dir +str(self.data[idx][1])+ '/' + str(self.data[idx][0]) + '.jpg'
        img = read_img(path)
        img = cvImg_to_torch(img).float()
        label = torch.tensor([self.data[idx][1]]).int()
        if self.transform :
            img = self.transform(img)
        return img,label
 
class ClassficationNet_block_16(nn.Module):
    """
    网络类
    """
    def __init__(self):
        """
        初始化网络
        """
        super(ClassficationNet_block_16, self).__init__()
        
        self.r1 = nn.ReflectionPad2d(1)
        # [16*16 --> 8*8] 
        self.c1 = nn.Conv2d(3, 16, kernel_size=4, padding=0, stride=2)
        # [8*8 --> 4*4] 
        self.c2 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)
        # [4*4 --> 1*1]
        self.c3 = nn.Conv2d(32, 64, kernel_size=4, padding=0, stride=1)
 
        # 后半部分全连间层
        self.l1 = nn.Linear(64,64)
        self.l2 = nn.Linear(64,1)
        self.s1 = nn.Sigmoid()
    
    def forward(self, input):
        """
        前向传播层
        参数：
            input：输入数据
        返回值：
            x：输出数据
        """
        x = self.r1(input)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = x.view(x.shape[0],-1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.s1(x)
        return x
 
    def predict(self,data):
        with torch.no_grad():
            output = (self.forward(data)>0.5).int()
            return output
 
class ClassficationNet_block_32(nn.Module):
    """
    网络类
    """
    def __init__(self):
        """
        初始化网络
        """
        super(ClassficationNet_block_32, self).__init__()
        self.r1 = nn.ReflectionPad2d(1)
        # [34*34 --> 16*16]
        self.c1 = nn.Conv2d(3, 8, kernel_size=4, padding=0, stride=2)
        # [16*16 --> 8*8] 
        self.c2 = nn.Conv2d(8, 16, kernel_size=4, padding=1, stride=2)
        # [8*8 --> 4*4] 
        self.c3 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)
        # [4*4 --> 1*1]
        self.c4 = nn.Conv2d(32, 64, kernel_size=4, padding=0, stride=1)
 
        # 后半部分全连间层
        self.l1 = nn.Linear(64,64)
        self.l2 = nn.Linear(64,1)
        self.s1 = nn.Sigmoid()
    
    def forward(self, input):
        """
        前向传播层
        参数：
            input：输入数据
        返回值：
            x：输出数据
        """
        x = self.r1(input)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = x.view(x.shape[0],-1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.s1(x)
        return x
 
    def predict(self,data):
        with torch.no_grad():
            output = (self.forward(data)>0.5).int()
            return output
 
class ClassficationNet_pixel(nn.Module):
    """
    网络类
    """
    def __init__(self):
        """
        初始化网络
        """
        super(ClassficationNet_pixel, self).__init__()
 
        # [32*32 --> 34*34]
        self.r1 = nn.ReflectionPad2d(1)
        # [34*34 --> 16*16]
        self.c1 = nn.Conv2d(3, 8, kernel_size=4, padding=0, stride=2)
        # [16*16 --> 8*8] 
        self.c2 = nn.Conv2d(8, 16, kernel_size=4, padding=1, stride=2)
        # [8*8 --> 4*4] 
        self.c3 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)
        # [4*4 --> 1*1]
        self.c4 = nn.Conv2d(32, 64, kernel_size=4, padding=0, stride=1)
 
        # 后半部分全连间层
        self.l1 = nn.Linear(64,64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,1)
        self.s1 = nn.Sigmoid()
    
    def forward(self, input):
        """
        前向传播层
        参数：
            input：输入数据
        返回值：
            x：输出数据
        """
        x = self.r1(input)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = x.view(x.shape[0],-1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.s1(x)
        return x
 
    def predict(self,data):
        with torch.no_grad():
            output = (self.forward(data)>0.5).int()
            return output
