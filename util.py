import cv2
import numpy as np
import torch 
import time
from torch.utils.data import DataLoader
import torch.utils.data as data


class ImgDataset(data.Dataset): 
    def __init__(self, x,  y, folder, transform=None):
        self.x = x
        self.y = y
        self.folder = folder
        if y is not None:
            self.y = torch.LongTensor(y) 
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = cv2.imread(self.folder+str(self.x[index])+'.png', 0)
        if self.transform is not None:
            X = self.transform(X)
        Y = self.y[index]
        return X, Y

def GetLoader(X, y, batch_size, folder=None, transform=None, stage=0):
    X = X
    y = y
    train_set = ImgDataset(X, y, folder, transform)
    if stage:
        weights = []
        for i in range(len(y)):
            if y[i] == 0:
                weights.append(312)
            elif y[i] == 1:
                weights.append(344)
            elif y[i] == 2:
                weights.append(3131)
            elif y[i] == 3:
                weights.append(169)
        weights = np.asarray(weights)
        weights = weights.reshape(len(weights))
        weights = len(weights)/weights
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        loader = DataLoader(train_set, batch_size, pin_memory=True, sampler=sampler) #
    else:
        loader = DataLoader(train_set, batch_size, pin_memory=True)
    return loader
            
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def Train(net, num_epoch, train_data, test_data, optimizer, loss, schduler=None):
    best_acc = 0.75
    test_acc_log = []
    test_loss_log = []
    train_acc_log = []
    train_loss_log = []
    for epoch in range(num_epoch):
        init_time = time.time()
        train_acc = 0
        train_loss = 0
        test_acc = 0
        test_loss = 0
        conf_matrix_train = torch.zeros(net.num_classes, net.num_classes)
        conf_matrix_test = torch.zeros(net.num_classes, net.num_classes)
        net.train()
        for data, label in train_data:
            data = data.cuda()
            label = label.cuda()
            pred = net(data)
            optimizer.zero_grad()
            l = loss(pred, label)
            conf_matrix_train = confusion_matrix(pred, label, conf_matrix_train)
            conf_matrix_train = conf_matrix_train.cpu()
            train_loss += l
            train_acc += (pred.argmax(1)==label).sum().detach().cpu().numpy()/(len(label))
            l.backward()
            optimizer.step()
            if schduler:
                schduler.step()

        net.eval()
        with torch.no_grad():
            for data, label in test_data:
                data = data.cuda().float()
                label = label.cuda()
                #print('test data', data.shape)
                pred = net(data)
                #print('test pred', pred.shape)
                #print('test label', label.shape)
                l = loss(pred, label)
                test_acc += (pred.argmax(1)==label).sum().detach().cpu().numpy()/len(label)
                test_loss += l
                # 混淆矩阵
                conf_matrix_test = confusion_matrix(pred, label, conf_matrix_test)
                conf_matrix_test = conf_matrix_test.cpu()
        test_acc_log.append(test_acc/len(test_data))
        test_loss_log.append(test_loss.cpu().detach().numpy()/len(test_data))
        train_acc_log.append(train_acc/len(train_data))
        train_loss_log.append(train_loss.cpu().detach().numpy()/len(train_data))
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss/len(train_data):.3f} Valid Loss: {test_loss/len(test_data):.3f} Train Acc: {train_acc/len(train_data):.3f} Valid Acc: {test_acc/len(test_data):.3f} Epoch Time: {(time.time()-init_time):.3f}')
        conf_matrix_train = np.array(conf_matrix_train.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
        print('混淆矩阵train:')
        print(conf_matrix_train)
        conf_matrix_test = np.array(conf_matrix_test.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
        corrects_test = conf_matrix_test.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
        per_kinds_test = conf_matrix_test.sum(axis=1)  # 抽取每个分类数据总的测试条数
        print('混淆矩阵test:')
        print(conf_matrix_test)
        print("每种类别的识别准确率为：{0}".format([rate * 100 for rate in corrects_test / per_kinds_test]))
        if test_acc/len(test_data) > best_acc:
            best_acc = test_acc/len(test_data)
            torch.save(net.state_dict(), './save/' + f'{best_acc:.3f}.pt')
    return np.array([test_acc_log, test_loss_log, train_acc_log, train_loss_log])

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix
