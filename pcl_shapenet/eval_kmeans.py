'''
Script to generate embeddings from resnet trained using pcl
Command to run:

python eval_kmeans.py --pretrained experiment_pcl_resume/checkpoint.pth.tar /home/mprabhud/dataset/shapenet_renders/npys/

'''

from __future__ import print_function

import os
import sys
import time
import torch
import socket
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import random
import numpy as np
from tqdm import tqdm
import faiss
import numpy as np
import ipdb
st = ipdb.set_trace
import os
cwd = os.getcwdb().decode()
import sys
sys.path.append(cwd)
# st()

from torchvision import transforms, datasets
import torchvision.models as models
from pcl_shapenet.pcl import loader
from pcl_shapenet.eval_embeddings_imagenet import Feat_Embedding_Model



def parse_option():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser('argument for training')
    
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
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
    parser.add_argument('--low-dim', default=16, type=int,
                    help='feature dimension (default: 128)')
    parser.add_argument('--pcl-r', default=1024, type=int,
                        help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='softmax temperature')

    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco-v2/SimCLR data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    
    parser.add_argument('--imagenet', action='store_true',
                        help='use imagenet')    

    parser.add_argument('--num-cluster', default='100', type=str, 
                        help='number of clusters')

    opt = parser.parse_args()

    opt.num_class = 20
    
    # if low shot experiment, do 5 random runs
    if opt.low_shot:
        opt.n_run = 5
    else:
        opt.n_run = 1
    return opt




class PCL_kmeans(nn.Module):
    def __init__(self, args, pretrained_file = ""):
        super(PCL_kmeans, self).__init__()
        if args.imagenet:
            self.embedding_model = Feat_Embedding_Model(pretrained_file=pretrained_file)
        else:
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


    def compute_embeddings(self,eval_loader, args):
        print('Computing embeddings...')
        self.embedding_model.eval()
        # st()
        features = torch.zeros(len(eval_loader.dataset),16).cuda()
        for i, (images, index) in enumerate(tqdm(eval_loader)):
            with torch.no_grad():
                if args.imagenet:
                    pass
                else:
                    images = images + 0.5    
                    #images in range of 0-1
                    t_images  = []                
                    for image_ex in images:
                        image_ex = self.normalize(image_ex)
                        t_images.append(image_ex)                            
                    images = torch.stack(t_images)
                images = images.cuda(non_blocking=True)
                st()
                feat = self.embedding_model(images) 
                features[index] = feat    
        return features.cpu()


    def run_kmeans(self,x, args):
        """
        Args:
            x: data to be clustered
        """
        results = {'im2cluster':[],'centroids':[],'density':[]}
        
        for seed, num_cluster in enumerate(args.num_cluster):
            print('performing kmeans clustering on ...',num_cluster)
            # intialize faiss clustering parameters
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 1000
            clus.min_points_per_centroid = 10

            res = faiss.StandardGpuResources()

            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = 0   
            index = faiss.GpuIndexFlatL2(res, d, cfg)  

            clus.train(x, index)   

            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]
            
            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
            
            # sample-to-centroid distances for each cluster 
            Dcluster = [[] for c in range(k)]          
            for im,i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])
            
            # concentration estimation (phi)        
            density = np.zeros(k)
            for i,dist in enumerate(Dcluster):
                if len(dist)>1:
                    d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                    density[i] = d     
                    
            #if cluster only has one point, use the max to estimate its concentration        
            dmax = density.max()
            for i,dist in enumerate(Dcluster):
                if len(dist)<=1:
                    density[i] = dmax 

            density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
            density = args.temperature*density/density.mean()  #scale the mean to temperature 
            
            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids = nn.functional.normalize(centroids, p=2, dim=1)    

            im2cluster = torch.LongTensor(im2cluster).cuda()               
            density = torch.Tensor(density).cuda()
            
            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)    
        return results


    def forward(self, loader, args):
        cluster_result = None
        # st()
        features = self.compute_embeddings(loader,args) 
        # placeholder for clustering result
        cluster_result = {'im2cluster':[],'centroids':[],'density':[]}

        for num_cluster in args.num_cluster:
            cluster_result['im2cluster'].append(torch.zeros(len(loader),dtype=torch.long).cuda())
            cluster_result['centroids'].append(torch.zeros(int(num_cluster),16).cuda())
            cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
            
        # features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
        features = features.numpy()
        cluster_result = self.run_kmeans(features,args)  #run kmeans clustering 
        return cluster_result


def main():
    args = parse_option()
    args.num_cluster = args.num_cluster.split(',')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    ########################################################################
    # STEP 1: SETuP DATALOADER (MAKE SURE TO CONVERT IT TO PIL IMAGE !!!!!)#
    ########################################################################
    hostname = socket.gethostname()
    
    if "compute" in hostname:
        traindir = "/home/mprabhud/dataset/shapenet_renders/npys/"
    else:
        traindir = '/media/mihir/dataset/shapenet_renders/npys/'

    
    train_dataset = loader.ShapeNet(
        traindir,
        'split_allpt.txt')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*2, shuffle=False,
        sampler=None, num_workers=args.num_workers, pin_memory=True)



    if "compute" in hostname:
        if args.imagenet:
            pretrained_file = "/home/mprabhud/ishita/other/resnet/18oct_frozen_2048_deepAutoencmodel_best.pth.tar"
        else:
            pretrained_file = "/home/mprabhud/ishita/pcl-shapenet/experiment_pcl_resume/checkpoint.pth.tar"
    else:
        pretrained_file = ""


    pcl_kmeans = PCL_kmeans(args,pretrained_file = pretrained_file)

    cluster_result = pcl_kmeans(train_loader,args)

    cluster_result_centroid = cluster_result['centroids'][0]
    # st()
    if args.imagenet:
        np.save("pcl_shapenet/dump/kmeans_imagenet.npy",cluster_result_centroid.cpu().numpy())
    else:
        np.save("pcl_shapenet/dump/kmeans_pcl.npy",cluster_result_centroid.cpu().numpy())
    # print("check")
    
    ############################
    # STEP 2: INITIALIZE MODEL #
    ############################

    # create model
    
if __name__ == '__main__':
    main()