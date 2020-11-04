
import argparse
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
import ipdb
st = ipdb.set_trace
import logging 
import socket
from pcl_shapenet.pcl import loader
from tqdm import tqdm
from tensorboardX import SummaryWriter



import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


class Test_Model(nn.Module):
	def __init__(self):
		super(Test_Model, self).__init__()
		self.base_model = models.resnet18(pretrained=True)
		self.base_model.fc = nn.Linear(512, 2048)
		self.relu1 = nn.LeakyReLU()
		self.relu2 = nn.LeakyReLU()
		self.relu3 = nn.LeakyReLU()
		self.relu4 = nn.LeakyReLU()
		self.relu5 = nn.LeakyReLU()
		self.relu6 = nn.LeakyReLU()
		self.relu7 = nn.LeakyReLU()
		# self.relu8 = nn.LeakyReLU(inplace=True)
		# self.relu9 = nn.LeakyReLU(inplace=True)
		self.fc1 = nn.Linear(2048, 1024)
		self.fc2 = nn.Linear(1024, 256)
		self.fc3 = nn.Linear(256, 16)
		# self.fc4 = nn.Linear(64, 16)
		# self.fc5 = nn.Linear(16, 64)
		self.fc6 = nn.Linear(16, 256)
		self.fc7 = nn.Linear(256, 1024)
		self.fc8 = nn.Linear(1024, 2048)
		self.fc9 = nn.Linear(2048, 1000)


	def forward(self,x):
		x = self.relu1(self.base_model(x))
		x = self.relu2(self.fc1(x))
		x = self.relu3(self.fc2(x))
		x = self.fc3(x)
		# x = self.relu5(self.fc4(x))
		# x = self.relu6(self.fc5(x))
		# x = self.relu5(self.fc6(x))
		# x = self.relu6(self.fc7(x))
		# x = self.relu7(self.fc8(x))
		# x = self.fc9(x)
		return x


class Feat_Embedding_Model(nn.Module):

	'''
	freeze_wts: True, if MLP should also be frozen, else False
	path: path to checkpoint.pth.tar file
	'''
	def __init__(self, pretrained_file=None):
		super(Feat_Embedding_Model, self).__init__()
		model_full = Test_Model()

		if pretrained_file != "":
			print("loaded "+ pretrained_file)
			checkpoint = torch.load(os.path.join(pretrained_file))
			model_full.load_state_dict(checkpoint['state_dict'])


		self.model = nn.Sequential(*list(model_full.children())[:-4])



	def forward(self, x):
		x = x + 0.5
		return self.model(x)





def main():
    # args = parse_option()
    
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    ########################################################################
    # STEP 1: SETuP DATALOADER (MAKE SURE TO CONVERT IT TO PIL IMAGE !!!!!)#
    ########################################################################

    # traindir = os.path.join(args.data)

    hostname = socket.gethostname()

    if "compute" in hostname:
        traindir = "/home/mprabhud/dataset/shapenet_renders/npys/"
        pretrained_file = "/home/mprabhud/ishita/pcl-shapenet/experiment_pcl_resume/checkpoint.pth.tar"
    else:
        traindir = '/media/mihir/dataset/shapenet_renders/npys/'
        pretrained_file = ""

    train_dataset = loader.ShapeNet(
        traindir,
        'split_allpt.txt',)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False,
        sampler=None, num_workers=0, pin_memory=True)


    imagenet_embed = Feat_Embedding_Model(pretrained_file=pretrained_file)

    for i, (images, index) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            images = images.cuda()
            feat = imagenet_embed(images)
    

    
if __name__ == '__main__':
    main()