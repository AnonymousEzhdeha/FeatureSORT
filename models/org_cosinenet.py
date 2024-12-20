import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import numpy as np
import torch
# from torch2trt import torch2trt
import cv2
import PIL.Image as pil_img
import torchvision


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True)

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class OriginalCosineNet(nn.Module):
    def __init__(self, num_classes=751 ,reid=False): # for cosine net original paper, the class number is 751
        super(OriginalCosineNet,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64,64,2,False)
        # 32 64 32
        self.layer2 = make_layers(64,128,2,True)
        # 64 32 16
        self.layer3 = make_layers(128,256,2,True)
        # 128 16 8
        self.layer4 = make_layers(256,512,2,True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 256 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 512
        
        if self.reid:
            # x = x.div(x.norm(p=2,dim=1,keepdim=True))
            x = torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12, out=None) #This layer only is convertible into
            # torch2trt engine, but NOT into wts file. It means that if we use wts file for constructing engine, we must 
            # implement this layer manually inside cpp file, since there is no weight assigned to this layer. However, if 
            # we use torch2trt engine it would consider it (maybe it has built in function to convert this layer to engine as well)
            return x
        # classifier
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = OriginalCosineNet(reid=True)
    state_dict = torch.load("/media/ezhdeha/B2CA8AACCA8A6C83/Pintel_Projects/DeepLearning/phrd_cosinenet_train/org_cosinenet.t7", map_location=lambda storage, loc: storage)['net_dict']
    net.load_state_dict(state_dict)
    net.to('cuda').eval()
    x = torch.ones(1,3,128,64).cuda()
    xx = torch.zeros((0, 3, 128, 64))
    y = net(x)
    a=2
    ##

