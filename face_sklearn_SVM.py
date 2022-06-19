"""
train and test, LBP+PCA+Bayes
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
    X = cv2.imread(f_path,0)
    X = X.reshape(128*128)
    train_data_image.append(X)

test_data_image = []
for fileName in test_data_list:
    f_path = './png/' + fileName + '.png'
    X = cv2.imread(f_path,0)
    X = X.reshape(128*128)
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
plt.matshow(CM, cmap=plt.cm.Reds)
#plt.imshow(Z)
#plt.colorbar()
#plt.show()

print('Grey+SVM') 

num3 = 0
for i in Y:
    if i == 3:
        num3 = num3+1