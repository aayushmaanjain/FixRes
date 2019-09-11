"""
1. Clone repository
2. [Skip] ~Provide ImageNet Path on line 99~
3. Run script with Python3.6 (that's the one I ran with): $ python3.6 resnext.py

Docker Image: rocm2.7_ubuntu16.04_py3.6_pytorch
"""
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from imnet_finetune.resnext_wsl import *
from imnet_finetune.transforms import get_transforms

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.epoch_sum = 0
        self.epoch_count = 0
        self.epoch_avg = 0

    def reset(self):
        # self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.epoch_sum += val * n
        self.epoch_count += n
        self.epoch_avg = self.epoch_sum / self.epoch_count
                                                                    
    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '} ({epoch_avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
     """Computes the accuracy over the k top predictions for the specified values of k"""
     with torch.no_grad():
         maxk = max(topk)
         batch_size = target.size(0)
             
         _, pred = output.topk(maxk, 1, True, True)
         pred = pred.t()
         correct = pred.eq(target.view(1, -1).expand_as(pred))

         res = []
         for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
         return res


def main():    
    model = resnext101_32x48d_wsl(progress=False)
    # model = resnext101_32x8d_wsl(progress=False)

    ## Hyper-parametes
    batch_size = 2  # for resnext101_32x48d  
    # batch_size = 28   # for resnext101_32x8d

    ## Misc configs
    print_freq = 1  # 500
    
    ## DataLoaders
    transformation = get_transforms(input_size=320, test_size=320, kind='full', crop=True, need=('train','val'), backbone=None)
    # trainset = torchvision.datasets.ImageFolder('/data/imnet/train', transform=transformation['val']) # ImageNet path
    trainset = torchvision.datasets.FakeData(size=6500, image_size=(3,224,224), num_classes=1000, transform=transformation['val'])
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=2)

    ## Model, Optimizer, Scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # lrscheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ## Training
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    for i, data in enumerate(trainloader):
        images, target = data
        print("Sending data to GPU", flush=True)
        images = images.to(device)
        target = target.to(device)

        print("Doing Forward pass", flush=True)
        output = model(images)
        # Compute losses
        print("Computing loss", flush=True)
        loss = criterion(output, target)
        # compute gradients and Backprop
        optimizer.zero_grad()
        print("Computing gradients", flush=True)
        loss.backward()
        print("Doing Backprop", flush=True)
        optimizer.step()
        # lrscheduler.step()

        acc1, acc5 = accuracy(output, target, (1,5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if i+1 % print_freq == 0:
            print(' * TRAIN: Acc@1 {top1.epoch_avg:.3f} Acc@5 {top5.epoch_avg:.3f}'.format(top1=top1, top5=top5))

if __name__ == '__main__':
    main()


