"""
train and test, LBP+PCA+Bayes
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import myHOG

# get data and separate the training and testing data
total_data_list = np.load('./shuffled_data.npy')
total_labels_list = np.load('./shuffled_label.npy')

train_data_list = total_data_list[0:3000]
test_data_list = total_data_list[3000:3957]
train_labels_list = total_labels_list[0:3000]
test_labels_list = total_labels_list[3000:3957]

# read image
train_data_image = []
for fileName in train_data_list:
    f_path = './png/' + fileName + '.png'
    X = cv2.imread(f_path,cv2.IMREAD_GRAYSCALE)
    #X = X.reshape(128*128)
    
    hog = myHOG.Hog_descriptor(X, cell_size=2, bin_size=1)
    vector, image = hog.extract()
    X = image.reshape(128*128)
    
    train_data_image.append(X)

test_data_image = []
for fileName in test_data_list:
    f_path = './png/' + fileName + '.png'
    X = cv2.imread(f_path,cv2.IMREAD_GRAYSCALE)
    #X = X.reshape(128*128)
    
    hog = myHOG.Hog_descriptor(X, cell_size=2, bin_size=1)
    vector, image = hog.extract()
    X = image.reshape(128*128)
    
    test_data_image.append(X)
    
# create PCA
pca = PCA(n_components=100)
pca.fit(train_data_image)
X = pca.transform(train_data_image)
Y = train_labels_list.reshape(len(train_labels_list))

# SVM
clf = SVC()
clf.fit(X, Y)

# test
X = pca.transform(test_data_image)
Y = test_labels_list.reshape(len(test_labels_list))
Y_predict = clf.predict(X)

# recall
CM = confusion_matrix(Y, Y_predict)
#plt.matshow(CM, cmap=plt.cm.Reds)
#plt.imshow(Z)
#plt.colorbar()
#plt.show()

cm = CM
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
plt.colorbar()
labels_name = ['child','teen','adult','senior']
num_local = np.array(range(len(labels_name)))  
plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
plt.ylabel('True label')    
plt.xlabel('Predicted label')
for i in range(4):
   for j in range(4):
       data = cm[j][i]
       data = str(data)[0:5]
       plt.text(-0.4+i, 0.1+j, data, fontsize='x-large',color = 'red')
