import torch.nn as nn
import torch.nn.functional as F


class VGG16_torch(nn.Module):
    def __init__(self):
        super(VGG16_torch, self).__init__()
        self.num_classes = 4
        # GROUP 1
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.maxpool1 = nn.MaxPool2d(2)  # 池化后长宽减半 output:16*16*64

        # GROUP 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.maxpool2 = nn.MaxPool2d(2)  # 池化后长宽减半 output:8*8*128

        # GROUP 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)  # output:8*8*256
        self.maxpool3 = nn.MaxPool2d(2)  # 池化后长宽减半 output:4*4*256

        # GROUP 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)  # output:4*4*512
        self.maxpool4 = nn.MaxPool2d(2)  # 池化后长宽减半 output:2*2*512

        # GROUP 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)  # output:14*14*512
        self.maxpool5 = nn.MaxPool2d(2)  # 池化后长宽减半 output:1*1*512

        self.fc1 = nn.Linear(in_features=8192, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=4)

    # 定义前向传播
    def forward(self, x):
        input_dimen = x.size(0)

        # GROUP 1
        output = self.conv1_1(x)
        output = F.relu(output)
        output = self.conv1_2(output)
        output = F.relu(output)
        output = self.maxpool1(output)

        # GROUP 2
        output = self.conv2_1(output)
        output = F.relu(output)
        output = self.conv2_2(output)
        output = F.relu(output)
        output = self.maxpool2(output)

        # GROUP 3
        output = self.conv3_1(output)
        output = F.relu(output)
        output = self.conv3_2(output)
        output = F.relu(output)
        output = self.conv3_3(output)
        output = F.relu(output)
        output = self.maxpool3(output)

        # GROUP 4
        output = self.conv4_1(output)
        output = F.relu(output)
        output = self.conv4_2(output)
        output = F.relu(output)
        output = self.conv4_3(output)
        output = F.relu(output)
        output = self.maxpool4(output)

        # GROUP 5
        output = self.conv5_1(output)
        output = F.relu(output)
        output = self.conv5_2(output)
        output = F.relu(output)
        output = self.conv5_3(output)
        output = F.relu(output)
        output = self.maxpool5(output)

        output = output.view(x.size(0), -1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        # 返回输出
        return output

