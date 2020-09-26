"""
函数：
    read_image：读取图像
    show_image：展示图像
    save_img：保存图像
    cvImg_to_pltImg：opencv图像转化为plt图像
    cvImg_to_torch：torch数组转换为opencv图像
    torch_to_cvImg：opencv图像转换为torch数组
    
"""
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from torch.utils.data import DataLoader,Dataset

def read_img(path):
    img = cv.imread(path)/255
    if type(img) == None:
        print("未读取到图片")
    return img 

def show_img(imgs):
    """
    展示一张或多张opencv格式图片。
    参数：
        images：图片或图片列表或图片数组。
    """
    if not isinstance(imgs,list):
        if not isinstance(imgs,tuple):
            imgs = [imgs]
    for i in imgs:
        i32 = i.astype('float32')
        I = cvImg_to_pltImg(i32)
        plt.imshow(I)
        plt.show()

def save_img(img,path):
    """
    保存图片，失败会提示
    """
    isSave = cv.imwrite(path,img*255)
    if not isSave:
        print("Failed to save") 

def cvImg_to_pltImg(img):
    """
    将opencv类型图片转换为plt类型图片。
    """
    I = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    return I

def cvImg_to_torch(img):
    """
    将opencv格式图像转化为torch类型。
    """
    I = torch.from_numpy(np.transpose(img,(2,0,1)))
    return I

def torch_to_cvImg(img):
    """
    将torch转换为opencv格式图像。
    """
    I = img.numpy().transpose((1,2,0))
    return I

