'''
Script to generate embeddings from resnet trained using pcl
Command to run:

python eval_embeddings.py --pretrained experiment_pcl_resume/checkpoint.pth.tar /home/mprabhud/dataset/shapenet_renders/npys/

'''

from __future__ import print_function

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import random
import numpy as np
from tqdm import tqdm
import socket
from torchvision import transforms, datasets
import torchvision.models as models
from pcl_shapenet.pcl import loader

import ipdb
st = ipdb.set_trace


def parse_option():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser('argument for training')
    
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--cost', type=str, default='0.5')
    parser.add_argument('--seed', default=0, type=int)
    
    # model definition
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')    
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained checkpoint')
    # dataset
    parser.add_argument('--low-shot', default=False, action='store_true', help='whether to perform low-shot training.')    
    
    opt = parser.parse_args()

    opt.num_class = 20
    
    # if low shot experiment, do 5 random runs
    if opt.low_shot:
        opt.n_run = 5
    else:
        opt.n_run = 1
    return opt

class PCL_embed(nn.Module):
    def __init__(self, pretrained_file = ""):
        super(PCL_embed, self).__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])    
        model_arch = 'resnet50'
        print("=> creating model '{}'".format(model_arch))
        self.embedding_model = models.__dict__[model_arch](num_classes=16)
        self.embedding_model.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), self.embedding_model.fc)
        
        if os.path.isfile(pretrained_file):
            print("=> loading checkpoint '{}'".format(pretrained_file))
            checkpoint = torch.load(pretrained_file, map_location="cpu")            
            state_dict = checkpoint['state_dict']
            # rename pre-trained keys
            
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q'): 
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]  
            if len(state_dict.keys()) == 0:
                assert False
                st()

            self.embedding_model.load_state_dict(state_dict, strict=False)        
        else:
            print("=> no checkpoint found at '{}'".format(pretrained_file))
        self.embedding_model.cuda()

    def forward(self, images):
        images = images + 0.5
        t_images  = []
        for image_ex in images:
            image_ex = self.normalize(image_ex)
            t_images.append(image_ex)
        images = torch.stack(t_images)
        feat = self.embedding_model(images) 
        return feat

def main():
    args = parse_option()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    ########################################################################
    # STEP 1: SETuP DATALOADER (MAKE SURE TO CONVERT IT TO PIL IMAGE !!!!!)#
    ########################################################################

    # traindir = os.path.join(args.data)

    hostname = socket.gethostname()

    if "compute" in hostname:
        traindir = "/home/mprabhud/dataset/shapenet_renders/npys/"
    else:
        traindir = '/media/mihir/dataset/shapenet_renders/npys/'


    train_dataset = loader.ShapeNet(
        traindir,
        'split_allpt.txt',)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=None, num_workers=0, pin_memory=True)


    pcl_embed = PCL_embed(pretrained_file = args.pretrained)

    for i, (images, index) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            images = images.cuda()
            feat = pcl_embed(images)

    ############################
    # STEP 2: INITIALIZE MODEL #
    ############################

    # create model
    # load from pre-trained

    
    
    ##########################
    # STEP 3: GET EMBEDDINGS #
    ##########################
    

    
if __name__ == '__main__':
    main()