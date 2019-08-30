#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('/opt/rocm/bin/rocm-smi')


# In[2]:


from collections import OrderedDict
# from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# In[3]:


def get_train_val_loaders(data_dir, batch_size, val_ratio, shuffle=True, 
                          num_workers=0, sampler=SubsetRandomSampler, pin_memory=False):
  random_seed =1928
  # basic essential transformations
  normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
  
  train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(10),
                                        transforms.ToTensor(), 
                                        normalize])

#   test_transform = transforms.Compose([transforms.Resize((224, 224)),
#                                        transforms.ToTensor(),
#                                        normalize])
  
  trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
#   valset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)

  num_train = len(trainset)
  indices = list(range(num_train))
  split = int(np.floor(val_ratio * num_train))

  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

  train_idx, val_idx = indices[split:], indices[:split]
  print("Training set size:", len(train_idx), "Validation set size:", len(val_idx))
  trainsampler = sampler(train_idx)
  valsampler = sampler(val_idx)

  trainloader = DataLoader(trainset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers, pin_memory=pin_memory)
  valloader = DataLoader(trainset, batch_size=batch_size, sampler=valsampler, num_workers=num_workers, pin_memory=pin_memory)

  return (trainloader, valloader)


# trainloader, valloader = get_train_val_loaders('./data', batch_size, False, 4242, 0.2)


# In[4]:


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
#     self.val = 0
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
  
def save_checkpoint(state, is_best, filename='checkpoint_conv.pth.tar'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, 'model_best_conv.pth.tar')
    
def imshow(img):
  unnormalize = transforms.Normalize((-0.4914/0.247, -0.4822/0.243, -0.4465/0.261), (1/0.247, 1/0.243, 1/0.261))
  img = unnormalize(img)
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


# In[5]:


# !apt  update
# !apt install wget
# !wget https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNeXt_101_32x48d.pth


# In[6]:


import torch
from imnet_evaluate.resnext_wsl import resnext101_32x48d_wsl

model = resnext101_32x48d_wsl(progress=True)

pretrained_dict = torch.load('ResNeXt_101_32x48d.pth', map_location='cpu')['model']

model_dict = model.state_dict()
for k in model_dict.keys():
  if(('module.'+k) in pretrained_dict.keys()):
    model_dict[k] = pretrained_dict.get(('module.'+k))
    
model.load_state_dict(model_dict)


# In[7]:


###################################################
## Settings
batch_size = 4
val_ratio = 10000/50000
batch_print_freq = 500
start_epoch = 0
# epochs = 1

###################################################
## Load Data
# dataloaders = {}
# dataloaders['train'], dataloaders['val'] = get_train_val_loaders('./data', batch_size, val_ratio)
trainloader, _ =  get_train_val_loaders('./data', batch_size, val_ratio)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# from imnet_finetune.transforms import get_transforms
# transformation = get_transforms(input_size=320,test_size=320, kind='full', crop=True, need=('train', 'val'), backbone=None)
# trainset = torchvision.datasets.ImageFolder('/workspace/data/train', transform=transformation['val'])
# trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=2)
# print(trainset)

###################################################
## Load Model
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define/load model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
# Send model to GPU
model.to(device)

# Define loss function (criterion) and optimizer and LR scheduler
criterion = nn.CrossEntropyLoss()  
# NOTE: define optimizer after sending model to GPU. May lead to error otherwise.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
#   lrscheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[ ]:


## Profiling Training on GPU
losses = AverageMeter('Loss', ':.4e')
top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')

# set to train mode
model.train()

trainiter = iter(trainloader)
# specify which batch you want to profile
batches = 1
isProfile = False
for i in range(batches):
    images, target = trainiter.next()
    images = images.to(device)
    target = target.to(device)
    print("data loaded")
#     if i == (batches-1):
#         isProfile = True
    
#     with torch.autograd.profiler.profile(enabled=isProfile,use_cuda=True) as prof:
    output = model(images)
    print("output done")
    loss = criterion(output, target)
    print("loss done")
  # compute gradients and do kprop 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("backprop done")
  
   # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), images.size(0))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))
    
    print(' * TRAIN: Acc@1 {top1.epoch_avg:.3f} Acc@5 {top5.epoch_avg:.3f}'.format(top1=top1, top5=top5))
    




