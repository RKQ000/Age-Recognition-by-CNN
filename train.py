# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:22:52 2022

@author: RocketQI
"""

import os
import numpy as np
import cv2
import torch 
from util import *
from VGG import *
from ResNet import *
from RegNet import *
from DenseNet import *
from MobileNetV2 import *
from EfficientNetV2 import *
from torchvision.transforms import transforms
import torch.optim as optim

cuda = 0

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(64),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomCrop(64),
    transforms.ToTensor(),
])

# train_labels = np.load('./train_lable.npy')
# valid_labels = np.load('./test_lable.npy')
# train_list = np.load('./train_data.npy')
# valid_list = np.load('./test_data.npy')

data = np.load('./shuffled_data.npy')
labels = np.load('./shuffled_label.npy')

train_list = data[0:3000]
valid_list = data[3000:3956]
train_labels = labels[0:3000]
valid_labels = labels[3000:3956]

train_loader = GetLoader(X=train_list, y=train_labels, batch_size=16, folder='./png/', transform=train_transform, stage=1)
valid_loader = GetLoader(X=valid_list, y=valid_labels, batch_size=16, folder='./png/', transform=valid_transform, stage=0)

#model = VGG16_torch().cuda()
#model = myNet().cuda()
#model = ResNet(ResBlock).cuda()
#model = MobileNetV2().cuda()
#model = create_regnet(model_name='RegNetY_400MF', num_classes=4).to('cuda:0')
#model = densenet121(pretrained=False).cuda()
model = EfficientNet().cuda()

total = get_n_params(model)
print('The number of parameters: ', total)
loss1 = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
schduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[39, 50], gamma=0.1)
print('Start Training')
test_acc_log, test_loss_log, train_acc_log, train_loss_log = Train(model, 70, train_loader, valid_loader, optimizer, loss1, schduler)
np.save('test_acc_log.npy', test_acc_log)
np.save('test_loss_log.npy', test_loss_log)
np.save('train_acc_log.npy', train_acc_log)
np.save('train_loss_log.npy', train_loss_log)
torch.save(model.state_dict(), './save/' + 'xxx_latest.pt')

