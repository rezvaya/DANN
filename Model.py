import torch.nn as nn
from torch.autograd import Function
import torch.utils.data as data
from PIL import Image
import random
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import numpy as np
from torchvision import datasets, models, transforms

def simple_conv_block(in_channels, 
                      out_channels, 
                      kernel_size, 
                      stride, 
                      padding,
                      pool_size, 
                      pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_size, pool_stride))

def get_target_classifier():
    clf = nn.Sequential()
    clf.add_module('c_fc1', nn.Linear(81920,32))
    clf.add_module('c_bn1', nn.BatchNorm1d(32))
    clf.add_module('c_relu1', nn.ReLU(True))
    clf.add_module('c_drop1', nn.Dropout2d())
    clf.add_module('c_fc2', nn.Linear(32, 32))
    clf.add_module('c_bn2', nn.BatchNorm1d(32))
    clf.add_module('c_relu2', nn.ReLU(True))
    clf.add_module('c_fc3', nn.Linear(32, 7))
    clf.add_module('c_softmax', nn.LogSoftmax(dim=1))
    return clf

def get_domain_classifier():
    domain_clf = nn.Sequential()
    domain_clf.add_module('d_fc1', nn.Linear(81920,32))
    domain_clf.add_module('d_bn1', nn.BatchNorm1d(32))
    domain_clf.add_module('d_relu1', nn.ReLU(True))
    domain_clf.add_module('d_fc2', nn.Linear(32, 2))
    domain_clf.add_module('d_softmax', nn.LogSoftmax(dim=1))
    return domain_clf

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainTransferNet(nn.Module):
    def __init__(self, feature_extractor, classifier, domain_classifier):
        super(DomainTransferNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
        

    def forward(self, input_data, alpha):
        features = self.feature_extractor(input_data)
        features = features.view(features.size(0), -1)
        reverse_features = ReverseLayerF.apply(features, alpha)
        class_output = self.classifier(features)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output