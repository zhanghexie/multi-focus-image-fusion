from models import*
from net_tools import *
from img_tools import *
import time

def get_mask_block(Ia,Ib,net,size,paddingx=0,paddingy=0):

    def big_picture_process_block(images,net,size):
        imags = (images-0.5)*2
        mask = np.zeros((images.shape[2],images.shape[3]))
        for i in range(images.shape[2]//size):
            for j in range(images.shape[3]//size):
                temp = net(images[:,:,i*size:(i+1)*size,j*size:(j+1)*size])
                if temp[0]>temp[1]:
                    mask[i*size:(i+1)*size,j*size:(j+1)*size] = 1
        return mask

    h1,w1 = Ia.shape[0],Ia.shape[1]
    h,w = h1+paddingy,w1+paddingx
    if h % size != 0:
        h = (h//size+1)*size
    if w % size != 0:
        w = (w//size+1)*size

    imgs = torch.zeros((2,Ia.shape[2],h1,w1)).to(device)
    imgs[0] = cvImg_to_torch(Ia)
    imgs[1] = cvImg_to_torch(Ib)
    imgs = nn.ReflectionPad2d((paddingy,w-w1-paddingy,paddingx,h-h1-paddingx))(imgs)
    temp = imgs[0].cpu()
    temp = temp.numpy().transpose((1,2,0))
    mask = big_picture_process_block(imgs,net,size)
    m = np.zeros_like(Ia)
    m[:,:,0] = mask[paddingx:paddingx+m.shape[0],paddingy:paddingy+m.shape[1]]
    m[:,:,1] = mask[paddingx:paddingx+m.shape[0],paddingy:paddingy+m.shape[1]]
    m[:,:,2] = mask[paddingx:paddingx+m.shape[0],paddingy:paddingy+m.shape[1]]
    return m

def get_mask_pixel(Ia,Ib,net):

    def big_picture_fast_process(images,net):
        """
        得到图片map的快速算法
        参数：
            images：两张图片
            trained_net：训练的网络
        返回值：
            返回map（代表每一点是否聚焦）
        """
        with torch.no_grad():
            outputs = torch.zeros(images.shape[0],1,images.shape[2],images.shape[3])
            images = (images-0.5)*2
            x = nn.ReflectionPad2d((16,15,16,15))(images)
            x = nn.ReflectionPad2d(1)(x)
            x = F.conv2d(x,weight=net.state_dict()['c1.weight'],bias=net.state_dict()['c1.bias'],padding=0,stride=1)
            x = F.conv2d(x,weight=net.state_dict()['c2.weight'],bias=net.state_dict()['c2.bias'],padding=2,stride=1,dilation=2)
            x = F.conv2d(x,weight=net.state_dict()['c3.weight'],bias=net.state_dict()['c3.bias'],padding=4,stride=1,dilation=4)
            x = F.conv2d(x,weight=net.state_dict()['c4.weight'],bias=net.state_dict()['c4.bias'],padding=0,stride=1,dilation=8)
            for i in range(outputs.shape[2]):
                for j in range(outputs.shape[3]):
                    temp = x[:,:,i,j]
                    temp = temp.view(temp.shape[0],-1)
                    temp = net.l1(temp)
                    temp = F.relu(temp)
                    temp = net.l2(temp)
                    temp = F.relu(temp)
                    temp = net.l3(temp)
                    temp = net.s1(temp)
                    outputs[:,:,i,j] = temp
            m = (outputs[0]-outputs[1]+1)/2
            m = m[0,:,:]
            return m
    imgs = torch.zeros((2,3,Ia.shape[0],Ia.shape[1])).to(device)
    imgs[0] = cvImg_to_torch(Ia)
    imgs[1] = cvImg_to_torch(Ib)
    mask = big_picture_fast_process(imgs,net)
    m = np.zeros((mask.shape[0],mask.shape[1],3))
    m[:,:,0] = mask
    m[:,:,1] = mask
    m[:,:,2] = mask
    return m