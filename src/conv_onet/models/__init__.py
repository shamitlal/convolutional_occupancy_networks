import torch
import ipdb
import pickle
st = ipdb.set_trace
import torch.nn as nn
import socket
import numpy as np
import torch.nn.functional as F
from torch import distributions as dist
from torch.nn.parameter import Parameter
from src.conv_onet.models import decoder
import torchvision.models as models
from pcl_shapenet.eval_embeddings_imagenet import  Feat_Embedding_Model
from src.encoder.pointnetpp import PointNetPlusPlusSSG
from src.encoder.pointnetpp import PointNetPlusPlusMSG

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_hyper': decoder.LocalDecoder_Hyper,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs,**kwargs)
        p_r = self.decode(p, c, **kwargs)
        return p_r


    def encode_inputs(self, inputs, **kwargs):
    # def encode_inputs(self, inputs,arg_dict=None):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs, **kwargs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        encodingnet = list(models.resnet18(pretrained=True).children())[:-1]
        self.encodingnet = nn.Sequential(*encodingnet).cuda()
        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.final_layer = nn.Linear(512, 16).cuda()
    def forward(self, input):
        # input -> B, 3, 128, 128  
        # label -> B
        encoding = self.encodingnet(input)
        encoding = self.flatten(self.leaky_relu(encoding))
        encoding = self.final_layer(encoding)
        # out = self.output_layer(encoding)
        # loss = self.criterion(out, label)
        return encoding



class HyperNet(nn.Module):
    def __init__(self,hypernet_params=None):
        super(HyperNet, self).__init__()
        self.emb_dimension = 16
        range_val = 1
        variance = ((2*range_val))/(12**0.5)
        lambda_val = 1
        variance = 0.2
        lambda_val = 0.5
        min_embed = -0.35
        max_embed = 0.6
        # vqvae_dict_size = 1000
        # st()
        self.vqvae_dict_size = hypernet_params['vqvae_dict_size']
        self.do_imagenet = hypernet_params['do_imagenet']

        # lambda_val = [lambda_val]*10 + [5,5]
        hostname = socket.gethostname()
        # st()
        if hypernet_params['use_rgb']:
            if self.do_imagenet:
                if "compute" in hostname:
                    pretrained_file = "/home/mprabhud/ishita/other/resnet/18oct_frozen_2048_deepAutoencmodel_best.pth.tar"
                else:
                    pretrained_file = ""  
                self.encodingnet = Feat_Embedding_Model(pretrained_file = pretrained_file)            
            else:              
                self.encodingnet = ResnetEncoder()
        else:
            self.encodingnet = PointNetPlusPlusMSG()

        self.embedding = nn.Embedding(self.vqvae_dict_size, self.emb_dimension)
        
        if self.do_imagenet:
            if "compute" in hostname:
                loaded_kmeans = torch.from_numpy(np.load("pcl_shapenet/dump/kmeans_imagenet_sdf_allt.npy"))
                self.embedding.weight.data = loaded_kmeans
        else:
            nn.init.normal_(self.embedding.weight, mean=0, std=0.4)

        self.prototype_usage = torch.zeros(self.vqvae_dict_size).cuda()
        # st()

        hypernet_names_shapes = pickle.load(open("hypernet.p","rb"))
        
        self.encoder_kernel_names = hypernet_names_shapes['encoder_kernel'][0]
        self.encoder_kernel_shape = hypernet_names_shapes['encoder_kernel'][1]

        self.encoder_bias_names = hypernet_names_shapes['encoder_bias'][0]
        self.encoder_bias_shape = hypernet_names_shapes['encoder_bias'][1]

        self.decoder_kernel_names = hypernet_names_shapes['decoder_kernel'][0]
        self.decoder_kernel_shape = hypernet_names_shapes['decoder_kernel'][1]

        self.decoder_bias_names = hypernet_names_shapes['decoder_bias'][0]
        self.decoder_bias_shape = hypernet_names_shapes['decoder_bias'][1]


        self.hidden1 = nn.Linear(16, 32)
        self.hidden2 = nn.Linear(32, 16)

        self.commitment_cost = 0.25

        encoder_weight_variances = [0.4, 0.055, 0.055, 0.08, 0.08, 0.055, 0.08, 0.055, 0.055, 0.08, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.15, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.15]
        encoder_weight_names = ['fc_pos.weight', 'blocks.0.fc_0.weight', 'blocks.0.fc_1.weight', 'blocks.0.shortcut.weight', 'blocks.1.fc_0.weight', 'blocks.1.fc_1.weight', 'blocks.1.shortcut.weight', 'blocks.2.fc_0.weight', 'blocks.2.fc_1.weight', 'blocks.2.shortcut.weight', 'blocks.3.fc_0.weight', 'blocks.3.fc_1.weight', 'blocks.3.shortcut.weight', 'blocks.4.fc_0.weight', 'blocks.4.fc_1.weight', 'blocks.4.shortcut.weight', 'fc_c.weight', 'unet3d.encoders.0.basic_module.SingleConv1.conv.weight', 'unet3d.encoders.0.basic_module.SingleConv2.conv.weight', 'unet3d.encoders.1.basic_module.SingleConv1.conv.weight', 'unet3d.encoders.1.basic_module.SingleConv2.conv.weight', 'unet3d.encoders.2.basic_module.SingleConv1.conv.weight', 'unet3d.encoders.2.basic_module.SingleConv2.conv.weight', 'unet3d.decoders.0.basic_module.SingleConv1.conv.weight', 'unet3d.decoders.0.basic_module.SingleConv2.conv.weight', 'unet3d.decoders.1.basic_module.SingleConv1.conv.weight', 'unet3d.decoders.1.basic_module.SingleConv2.conv.weight', 'unet3d.final_conv.weight']

        encoder_bias_variances = [0.5, 0.2, 0.2, 0.12, 0.13, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.1]
        encoder_bias_names = ['fc_pos.bias', 'blocks.0.fc_0.bias', 'blocks.0.fc_1.bias', 'blocks.1.fc_0.bias', 'blocks.1.fc_1.bias', 'blocks.2.fc_0.bias', 'blocks.2.fc_1.bias', 'blocks.3.fc_0.bias', 'blocks.3.fc_1.bias', 'blocks.4.fc_0.bias', 'blocks.4.fc_1.bias', 'fc_c.bias', 'unet3d.encoders.0.basic_module.SingleConv1.conv.bias', 'unet3d.encoders.0.basic_module.SingleConv2.conv.bias', 'unet3d.encoders.1.basic_module.SingleConv1.conv.bias', 'unet3d.encoders.1.basic_module.SingleConv2.conv.bias', 'unet3d.encoders.2.basic_module.SingleConv1.conv.bias', 'unet3d.encoders.2.basic_module.SingleConv2.conv.bias', 'unet3d.decoders.0.basic_module.SingleConv1.conv.bias', 'unet3d.decoders.0.basic_module.SingleConv2.conv.bias', 'unet3d.decoders.1.basic_module.SingleConv1.conv.bias', 'unet3d.decoders.1.basic_module.SingleConv2.conv.bias', 'unet3d.final_conv.bi1']

        decoder_weight_variances = [0.3, 0.13, 0.13, 0.13, 0.13, 0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.2]
        decoder_weight_names = ['fc_p.weight', 'fc_c.0.weight', 'fc_c.1.weight', 'fc_c.2.weight', 'fc_c.3.weight', 'fc_c.4.weight', 'blocks.0.fc_0.weight', 'blocks.0.fc_1.weight', 'blocks.1.fc_0.weight', 'blocks.1.fc_1.weight', 'blocks.2.fc_0.weight', 'blocks.2.fc_1.weight', 'blocks.3.fc_0.weight', 'blocks.3.fc_1.weight', 'blocks.4.fc_0.weight', 'blocks.4.fc_1.weight', 'fc_out.weight']

        decoder_bias_variances = [0.5,  0.1, 0.25, 0.07, 0.25, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]
        decoder_bias_names = ['fc_p.bias', 'fc_c.0.bias', 'fc_c.1.bias', 'fc_c.2.bias', 'fc_c.3.bias', 'fc_c.4.bias', 'blocks.0.fc_0.bias', 'blocks.0.fc_1.bias', 'blocks.1.fc_0.bias', 'blocks.1.fc_1.bias', 'blocks.2.fc_0.bias', 'blocks.2.fc_1.bias', 'blocks.3.fc_0.bias', 'blocks.3.fc_1.bias', 'blocks.4.fc_0.bias', 'blocks.4.fc_1.bias', 'fc_out.bias']
        self.decoder_kernelWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=decoder_weight_variances[index]), requires_grad=True) for index,i in enumerate(self.decoder_kernel_shape)])
        self.decoder_biasWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=decoder_bias_variances[index]), requires_grad=True) for index,i in enumerate(self.decoder_bias_shape)])

        self.encoder_kernelWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=encoder_weight_variances[index]), requires_grad=True) for index,i in enumerate(self.encoder_kernel_shape)])
        self.encoder_biasWeights =  nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=encoder_bias_variances[index]), requires_grad=True) for index,i in enumerate(self.encoder_bias_shape)])


        # st()
        # print("hello")


    def total(self,tensor_shape):
        tensor_shape = list(tensor_shape)
        prod = 1
        for i in tensor_shape:
            prod = prod*i
        return prod

    def update_embedding_dynamically(self):

        total = torch.sum(self.prototype_usage)
        probs = self.prototype_usage/total
        # Select 2 good embeds and take mean
        good_embeds_idxs = torch.where(probs>0.2)[0]
        if good_embeds_idxs.shape[0] < 2:
            print("Not enough highly used embedding. Skipping dynamic update.")
            return 

        random_idxs = torch.randperm(good_embeds_idxs.shape[0])[:2]
        good_embeds_idxs_random = good_embeds_idxs[random_idxs]
        good_embeds_random = self.embedding(good_embeds_idxs_random)
        good_embeds_random_detached = good_embeds_random.clone().detach()
        good_embeds_random_mean = torch.mean(good_embeds_random_detached, dim=0)

        # select 1 embed with 0 prob
        bad_embeds_idx = torch.where(probs <= 1e-7)[0]
        if bad_embeds_idx.shape[0] == 0:
            print("Not enough 0 used embedding. Skipping dynamic update.")
            return 

        random_idxs = torch.randperm(bad_embeds_idx.shape[0])[:1]
        bad_embeds_idxs_random = bad_embeds_idx[random_idxs]
        
        # replace embedding at bad_embeds_idxs_random with good_embeds_random_mean
        self.embedding.weight[bad_embeds_idxs_random] = good_embeds_random_mean

        self.prototype_usage *= 0 # clear usage

    def forward(self, input, arg_dict=None):
        loss = 0
        do_vis = 'logger' in arg_dict
        if do_vis:
            logger = arg_dict['logger']
            iteration = arg_dict['iteration']
            dynamic_dict = arg_dict['dynamic_dict']
        

        if do_vis:        
            if dynamic_dict:
                if (iteration%1000) ==0:
                    with torch.no_grad():
                        self.update_embedding_dynamically()
        # st()

        step_check = 250
        B = input.shape[0]
        # st()
        if arg_dict['rgb'] is None:
            embed = self.encodingnet(input)
        else:
            rgb = arg_dict['rgb']
            if self.do_imagenet:
                embed = self.encodingnet(rgb)
                embed = nn.functional.normalize(embed, p=2, dim=1)                    
            else:            
                embed = self.encodingnet(rgb)

        embed_shape = embed.shape            
        
        if 'logger' in arg_dict:        
            if (iteration%step_check) ==0:
                logger.add_histogram("embedding_generated", embed.clone().cpu().data.numpy(),iteration)    
                logger.add_histogram("embedding_init", self.embedding.weight.clone().cpu().data.numpy(),iteration)

        distances = (torch.sum(embed**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(embed, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        for idx in encoding_indices.view(-1):
            self.prototype_usage[idx] += 1

        if do_vis:        
            if (iteration%step_check) ==0:
                logger.add_histogram("embedding_indices_matched", encoding_indices.clone().cpu().data.numpy(),iteration)    

        encodings = torch.zeros(encoding_indices.shape[0], self.vqvae_dict_size, device=embed.device) 
        encodings.scatter_(1, encoding_indices, 1)    

        quantized = torch.matmul(encodings, self.embedding.weight).view(embed_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), embed)
        q_latent_loss = F.mse_loss(quantized, embed.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = embed + (quantized - embed).detach()
        embed = quantized

        embed = self.hidden1(embed)
        embed = F.leaky_relu(embed)
        embed = self.hidden2(embed)
        embed = F.leaky_relu(embed)
    
        if do_vis:                
            if (iteration%step_check) ==0:
                logger.add_histogram("embedding", embed.clone().cpu().data.numpy(),iteration)

        # st()
        encoder_kernels = [(torch.matmul(embed,self.encoder_kernelWeights[i])).view([B]+list(self.encoder_kernel_shape[i])) for i in range(len(self.encoder_kernelWeights))]
        encoder_bias = [(torch.matmul(embed,self.encoder_biasWeights[i])).view([B]+list(self.encoder_bias_shape[i])) for i in range(len(self.encoder_biasWeights))]

        decoder_kernels = [(torch.matmul(embed,self.decoder_kernelWeights[i])).view([B]+list(self.decoder_kernel_shape[i])) for i in range(len(self.decoder_kernelWeights))]
        decoder_bias = [(torch.matmul(embed,self.decoder_biasWeights[i])).view([B]+list(self.decoder_bias_shape[i])) for i in range(len(self.decoder_biasWeights))]

        if False:
            for ind,weight_name in enumerate(self.encoder_kernel_names):
                logger.add_histogram("encoder_weight_"+weight_name, encoder_kernels[ind].clone().cpu().data.numpy(),iteration)

            for ind,weight_name in enumerate(self.encoder_bias_names):
                logger.add_histogram("encoder_bias_"+weight_name, encoder_bias[ind].clone().cpu().data.numpy(),iteration)

            for ind,weight_name in enumerate(self.decoder_kernel_names):
                logger.add_histogram("decoder_weight_"+weight_name, decoder_kernels[ind].clone().cpu().data.numpy(),iteration)

            for ind,weight_name in enumerate(self.decoder_bias_names):
                logger.add_histogram("decoder_bias_"+weight_name, decoder_bias[ind].clone().cpu().data.numpy(),iteration)
        return [encoder_kernels,encoder_bias,decoder_kernels,decoder_bias], loss




class ConvolutionalOccupancyNetwork_Hypernet(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, device=None, hypernet_params= None, no_decoder=False):
        super().__init__()
        self.hypernet = HyperNet(hypernet_params = hypernet_params)
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None
        self.no_decoder = no_decoder
        # st()
        if not no_decoder:
            for name, param in self.decoder.named_parameters():
                print(name)
                assert False

        # st()
        for name, param in self.encoder.named_parameters():
            print(name)
            assert False

        self.decoder_wts = None
        self._device = device

    def forward(self, p, inputs, sample=True,rgb=None, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        # st()
        c_new = self.encode_inputs(inputs,**kwargs)

        if isinstance(c_new, list): 
            do_vqvae = True
            c,vqvae_loss = c_new
        else:
            c = c_new

        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs, **kwargs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''
        # st()
        arg_dict = kwargs['arg_dict']
        hyper_wts,loss = self.hypernet(inputs,arg_dict)
        encoder_wts = hyper_wts[:2]
        decoder_wts = hyper_wts[2:]
        self.decoder_wts = decoder_wts
        if self.encoder is not None:
            c = self.encoder(encoder_wts, inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)
        return [c, loss]

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        if self.no_decoder:
            logits = self.decoder(p, c, **kwargs)
        else:
            logits = self.decoder(self.decoder_wts, p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
