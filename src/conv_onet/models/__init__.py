import torch
import ipdb
import pickle
st = ipdb.set_trace
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch.nn.parameter import Parameter
from src.conv_onet.models import decoder
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
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
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
        if hyp.use_resnet_for_hypernet:
            self.encodingnet = models.resnet18().cuda()
            self.encodingnet.fc = nn.Linear(512, hyp.hypernet_input_size).cuda()
        else:
            activ = nn.LeakyReLU
            self.encodingnet = nn.Sequential(
                nn.Conv2d(3, 4, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(4, 8, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(8, 16, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(16, 32, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, hyp.hypernet_input_size),
                activ(),
                nn.Linear(hyp.hypernet_input_size, hyp.hypernet_input_size),
            ).cuda()
        self.output_layer = nn.Linear(hyp.hypernet_input_size, hyp.total_instances).cuda()  
        self.criterion = nn.CrossEntropyLoss()   

    def forward(self, input):
        # input -> B, 3, 128, 128  
        # label -> B
        # st()
        encoding = self.encodingnet(input)
        out = self.output_layer(encoding)
        # loss = self.criterion(out, label)
        return encoding



class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()
        self.emb_dimension = 16
        range_val = 1
        variance = ((2*range_val))/(12**0.5)
        lambda_val = 1
        variance = 0.2
        lambda_val = 0.5
        min_embed = -0.35
        max_embed = 0.6
        vqvae_dict_size = 100
        self.vqvae_dict_size = vqvae_dict_size

        # lambda_val = [lambda_val]*10 + [5,5]
        if False:
            self.encodingnet = ResnetEncoder()
        else:
            self.encodingnet = PointNetPlusPlusMSG()

        self.embedding = nn.Embedding(vqvae_dict_size, self.emb_dimension)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.6)
        self.prototype_usage = torch.zeros(vqvae_dict_size).cuda()
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

        encoder_weight_variances = [5.5e-2]*28
        encoder_bias_variances = [5.5e-2]*23

        decoder_weight_variances = [5e-2]*17
        decoder_bias_variances = [5e-2]*17


        self.encoder_kernelWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=encoder_weight_variances[index]), requires_grad=True) for index,i in enumerate(self.encoder_kernel_shape)])
        self.encoder_biasWeights =  nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=encoder_bias_variances[index]), requires_grad=True) for index,i in enumerate(self.encoder_bias_shape)])

        self.decoder_kernelWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=decoder_weight_variances[index]), requires_grad=True) for index,i in enumerate(self.decoder_kernel_shape)])
        self.decoder_biasWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=decoder_bias_variances[index]), requires_grad=True) for index,i in enumerate(self.decoder_bias_shape)])

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
        good_embeds_idxs = torch.where(probs>hyp.dynamic_hypernet_probability_thresh)[0]
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

    def forward(self, input):
        loss = 0
        # st()
        B = input.shape[0]
        embed = self.encodingnet(input)
        embed_shape = embed.shape            
        if False:
            summ_writer.summ_histogram("embedding_generated", embed.clone().cpu().data.numpy())    
            summ_writer.summ_histogram("embedding_init", self.embedding.weight.clone().cpu().data.numpy())

        distances = (torch.sum(embed**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(embed, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        for idx in encoding_indices.view(-1):
            self.prototype_usage[idx] += 1

        if False:
            summ_writer.summ_histogram("embedding_indices_matched", encoding_indices.clone().cpu().data.numpy())    

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

        if False:
            summ_writer.summ_histogram("embedding", embed.clone().cpu().data.numpy())    

        # st()
        encoder_kernels = [(torch.matmul(embed,self.encoder_kernelWeights[i])).view([B]+list(self.encoder_kernel_shape[i])) for i in range(len(self.encoder_kernelWeights))]
        encoder_bias = [(torch.matmul(embed,self.encoder_biasWeights[i])).view([B]+list(self.encoder_bias_shape[i])) for i in range(len(self.encoder_biasWeights))]

        decoder_kernels = [(torch.matmul(embed,self.decoder_kernelWeights[i])).view([B]+list(self.decoder_kernel_shape[i])) for i in range(len(self.decoder_kernelWeights))]
        decoder_bias = [(torch.matmul(embed,self.decoder_biasWeights[i])).view([B]+list(self.decoder_bias_shape[i])) for i in range(len(self.decoder_biasWeights))]
        if False:
            if hyp.vis_feat_weights:
                names = pickle.load(open('names.p',"rb"))
                for i in range(12):
                    weight_name = self.weight_names[i]+".weight"
                    bias_name = self.weight_names[i]+".bias"
                    summ_writer.summ_histogram(weight_name, feat_kernels[i].clone().cpu().data.numpy())
                    summ_writer.summ_histogram(bias_name, feat_Bias[i].clone().cpu().data.numpy())

        return [encoder_kernels,encoder_bias,decoder_kernels,decoder_bias], loss




class ConvolutionalOccupancyNetwork_Hypernet(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        self.hypernet = HyperNet()
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        # st()
        for name, param in self.decoder.named_parameters():
            print(name)
            assert False

        # st()
        for name, param in self.encoder.named_parameters():
            print(name)
            assert False

        self.decoder_wts = None
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
        st()
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''
        hyper_wts,loss = self.hypernet(inputs)
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
