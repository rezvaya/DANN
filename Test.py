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

from Data_loader import GetLoader


def testG(dataset_name, epoch):
    assert dataset_name in ['Source', 'RightPdd']

    model_root = 'models'
    image_root = os.path.join('./', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 256
    alpha = 0

    """load data"""

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

    if dataset_name == 'RightPdd':
        test_list = os.path.join(image_root, 'testPDD.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        test_list = os.path.join(image_root, 'testpVil.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'test'),
            data_list=test_list,
            transform=img_transform_target
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'PDD_transfer_net_epoch_' + str(epoch//10) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0


    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))