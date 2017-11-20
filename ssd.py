import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.autograd import Variable
from myloader import load_one_test


class ssd(nn.Module):
    def __init__(self):
        super(ssd, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)  #
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_fc6 = nn.Conv2d(512, 1024, 3, padding=6)
        self.conv_fc7 = nn.Conv2d(1024, 1024, 1)
        self.conv_fc7_conf = nn.Conv2d(1024, 126, 3, padding=1)  #
        self.conv_fc7_loc = nn.Conv2d(1024, 24, 3, padding=1)  #
        self.conv6_1 = nn.Conv2d(1024, 256, 1)
        self.conv6_2 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6_2_conf = nn.Conv2d(512, 126, 3, padding=1)
        self.conv6_2_loc = nn.Conv2d(512, 24, 3, padding=1)
        self.conv7_1 = nn.Conv2d(512, 128, 1)
        self.conv7_2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv7_2_conf = nn.Conv2d(256, 126, 3, padding=1)
        self.conv7_2_loc = nn.Conv2d(256, 24, 3, padding=1)
        self.conv8_1 = nn.Conv2d(256, 128, 1)
        self.conv8_2 = nn.Conv2d(128, 256, 3)
        self.conv8_2_conf = nn.Conv2d(256, 84, 3, padding=1)
        self.conv8_2_loc = nn.Conv2d(256, 16, 3, padding=1)
        self.conv9_1 = nn.Conv2d(256, 128, 1)
        self.conv9_2 = nn.Conv2d(128, 256, 3)
        self.conv9_2_conf = nn.Conv2d(256, 84, 3, padding=1)
        self.conv9_2_loc = nn.Conv2d(256, 16, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))  # conv1_1 & relu1_1
        x = F.relu(self.conv1_2(x))  # conv1_2 & relu1_2
        x = F.max_pool2d(x, 2)  # pool1
        x = F.relu(self.conv2_1(x))  # conv2_1 & relu2_1
        x = F.relu(self.conv2_2(x))  # conv2_2 & relu2_2
        x = F.max_pool2d(x, 2)  # pool2
        x = F.relu(self.conv3_1(x))  # relu3_1
        x = F.relu(self.conv3_2(x))  # relu3_2
        x = F.relu(self.conv3_3(x))  # relu3_3
        x = F.max_pool2d(x, 2)  # pool3
        x = F.relu(self.conv4_1(x))  # relu4_1
        x = F.relu(self.conv4_2(x))  # relu4_2
        x = F.relu(self.conv4_3(x))  # relu4_3
        x = F.max_pool2d(x, 2)  # pool4
        # conv4_3_norm=F.batch_norm(x,)#conv4_3_norm
        x = F.relu(self.conv5_1(x))  # relu5_1
        x = F.relu(self.conv5_2(x))  # relu5_2
        x = F.relu(self.conv5_3(x))  # relu5_3
        x = F.max_pool2d(x, 3, stride=1, padding=1)  # pool5
        x = F.relu(self.conv_fc6(x))  # relu6
        x = F.relu(self.conv_fc7(x))  # relu7
        x = F.relu(self.conv6_1(x))  # relu6_1
        x = F.relu(self.conv6_2(x))  # relu6_2
        x = F.relu(self.conv7_1(x))  # relu7_1
        # conv6_2_conf
        conv6_2_conf = self.conv6_2_conf(x).permute(2, 0, 1)
        conv6_2_conf = conv6_2_conf.view(-1, len(conv6_2_conf))
        # conv6_2_loc
        conv6_2_loc = self.conv6_2_loc(x).permute(2, 0, 1)
        conv6_2_loc = conv6_2_loc.view(-1, len(conv6_2_loc))
        x = F.relu(self.conv7_2(x))  # relu7_2
        # conv7_2_conf
        conv7_2_conf = self.conv7_2_conf(x).permute(2, 0, 1)
        conv7_2_conf = conv7_2_conf.view(-1, len(conv7_2_conf))
        # conv7_2_loc
        conv7_2_loc = self.conv7_2_loc(x).permute(2, 0, 1)
        conv7_2_loc = conv7_2_loc.view(-1, len(conv7_2_loc))
        x = F.relu(self.conv8_1(x))  # relu8_1
        x = F.relu(self.conv8_2(x))  # relu8_2
        # conv8_2_conf
        conv8_2_conf = self.conv8_2_conf(x).permute(2, 0, 1)
        conv8_2_conf = conv8_2_conf.view(-1, len(conv8_2_conf))
        # conv8_2_loc
        conv8_2_loc = self.conv8_2_loc(x).permute(2, 0, 1)
        conv8_2_loc = conv8_2_loc.view(-1, len(conv8_2_loc))
        x = F.relu(self.conv9_1(x))  # relu9_1
        x = F.relu(self.conv9_2(x))  # relu9_2
        # conv9_2_conf
        conv9_2_conf = self.conv9_2_conf(x).permute(2, 0, 1)
        conv9_2_conf = conv9_2_conf.view(-1, len(conv9_2_conf))
        # conv9_2_loc
        conv9_2_loc = self.conv9_2_loc(x).permute(2, 0, 1)
        conv9_2_loc = conv9_2_loc.view(-1, len(conv9_2_loc))
        loc = td.ConcatDataset([conv6_2_loc, conv7_2_loc, conv8_2_loc, conv9_2_loc])  # Concat mbox_loc
        conf = td.ConcatDataset([conv6_2_conf, conv7_2_conf, conv8_2_conf, conv9_2_conf])  # Concat mbox_conf_loc


net=ssd()
print net



print 'x'
