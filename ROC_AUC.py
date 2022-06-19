# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:45:39 2022

@author: RocketQI
"""
import torch 
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import cv2
from util import *
from ResNet import *
from RegNet import *
from DenseNet import *
from MobileNetV2 import *
from EfficientNetV2 import *
from torchvision.transforms import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
cuda = 0
valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomCrop(64),
    transforms.ToTensor(),
])

data = np.load('./shuffled_data.npy')
labels = np.load('./shuffled_label.npy')

valid_list = data[3000:3956]
valid_labels = labels[3000:3956]

valid_loader = GetLoader(X=valid_list, y=valid_labels, batch_size=16, folder='./png/', transform=valid_transform, stage=0)
model = EfficientNet().cuda()
model.load_state_dict(torch.load('./save/EfficientNet.pt'))

#  简单测试  --  ROC及AUC
scores = []
with torch.no_grad():
    for data, label in valid_loader:
        data = data.cuda().float()
        label = label.cuda()
        pred = model(data)
        
        scores += pred.cpu().detach().numpy().tolist()

scores = np.array(scores)
NUM_CLASSES = 4
binary_label = label_binarize(valid_labels, classes = list(range(NUM_CLASSES)))

fpr = {}
tpr = {}
roc_auc = {}

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(binary_label[:, i], scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(binary_label.ravel(), scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(NUM_CLASSES):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
mean_tpr /= NUM_CLASSES
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(8, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.savefig('Multi-class ROC.jpg', bbox_inches='tight')
plt.show()