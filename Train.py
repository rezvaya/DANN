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

from Test import testG
from Model import get_target_classifier, get_domain_classifier
from Model import DomainTransferNet
from Data_loader import GetLoader

source_dataset_name = 'Source'
target_dataset_name = 'RightPdd'
source_image_root = os.path.join('./', source_dataset_name)
target_image_root = os.path.join('./', target_dataset_name)
model_root = 'models'
cuda = True
cudnn.benchmark = True
lr = 0.0009
batch_size = 128
image_size = 256
n_epoch = 500
num_classes=7
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

feature_extractor = models.mobilenet_v2(pretrained=True).features
target_classifier=get_target_classifier()
domain_classifier=get_domain_classifier()

# load data
img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.4352, 0.5103, 0.2836], [0.2193, 0.2073, 0.2047])
])

img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.4352, 0.5103, 0.2836], [0.2193, 0.2073, 0.2047])   
])

train_list_source = os.path.join(source_image_root, 'trainpVil.txt')

dataset_source = GetLoader(
    data_root=os.path.join(source_image_root, 'train'),
    data_list=train_list_source,
    transform=img_transform_source
)

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

train_list = os.path.join(target_image_root, 'trainPDD.txt')

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'train'),
    data_list=train_list,
    transform=img_transform_target
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

# load model

transfer_net=DomainTransferNet(
    feature_extractor,
    target_classifier, 
    domain_classifier
    )

# setup optimizer

optimizer = optim.Adam(transfer_net.parameters(), lr=lr, eps=1e-08, amsgrad=True)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    transfer_net = transfer_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

ct = 0

for layer in feature_extractor.children():
  ct = ct + 1
  if ct < 12:
      for param in layer.parameters():
          param.requires_grad = False
  else:
      for param in layer.parameters():
          param.requires_grad = True

for p in transfer_net.classifier.parameters():
    p.requires_grad = True

for p in transfer_net.domain_classifier.parameters():
    p.requires_grad = True

# training

for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source),len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)
    i = 0

    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data

        data_source = data_source_iter.next()
        s_img, s_label = data_source

        transfer_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        
        class_output, domain_output = transfer_net(input_img, alpha)
        
        err_s_label = loss_class(class_output, class_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target
        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = transfer_net(input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label

        err.backward()
        optimizer.step()

        i += 1

        print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        
    number=epoch//10
    torch.save(transfer_net, '{0}/PDD_transfer_net_epoch_{1}.pth'.format(model_root, number))
    testG(source_dataset_name, epoch)
    testG(target_dataset_name, epoch)

print('done')