import torch
import torch.distributions as dist
from torch import nn
import pickle
import os
from src.encoder import encoder_dict
from src.conv_onet import models, training
from src.conv_onet import generation
from src import data
from src import config
from src.common import decide_total_volume_range, update_reso
from torchvision import transforms
import numpy as np
import ipdb
st = ipdb.set_trace

def get_model(cfg, device=None, dataset=None, logger=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    use_hypernet = cfg['model']['hypernet']
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']

    hypernet_params = cfg['model']['hypernet_params']
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    # update the feature volume/plane resolution
    if cfg['data']['input_type'] == 'pointcloud_crop':
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        if (dataset.split == 'train') or (cfg['generation']['sliding_window']):
            recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
            reso = cfg['data']['query_vol_size'] + recep_field - 1
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = update_reso(reso, dataset.depth)
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = update_reso(reso, dataset.depth)
        # if dataset.split == 'val': #TODO run validation in room level during training
        else:
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = dataset.total_reso
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = dataset.total_reso
    

    if use_hypernet:
        # st()
        if cfg['model']['hypernet_params']['no_decoder_hyper']:
            decoder = models.decoder_dict[decoder](
                dim=dim, c_dim=c_dim, padding=padding,
                **decoder_kwargs
            )            
        else:
            decoder = models.decoder_dict[decoder+"_hyper"](
                dim=dim, c_dim=c_dim, padding=padding,
                **decoder_kwargs
            )

        if encoder == 'idx':
            assert(False)
            encoder = nn.Embedding(len(dataset), c_dim)
        elif encoder is not None:
            encoder = encoder_dict[encoder+"_hyper"](
                dim=dim, c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
        else:
            assert(False)
            encoder = None

        model = models.ConvolutionalOccupancyNetwork_Hypernet(
            decoder, encoder, device=device, hypernet_params=hypernet_params, no_decoder=cfg['model']['hypernet_params']['no_decoder_hyper'],
        )

    else:
        decoder = models.decoder_dict[decoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **decoder_kwargs
        )

        if encoder == 'idx':
            encoder = nn.Embedding(len(dataset), c_dim)
        elif encoder is not None:
            encoder = encoder_dict[encoder](
                dim=dim, c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
        else:
            encoder = None

        model = models.ConvolutionalOccupancyNetwork(
            decoder, encoder, device=device
        )
    
    if cfg['model']['vis_weights']:
        encoder_kernel_names = []
        encoder_bias_names = []
        encoder_kernel_shapes = []
        encoder_bias_shapes = []        

        decoder_kernel_names = []
        decoder_bias_names = []
        decoder_kernel_shapes = []
        decoder_bias_shapes = []

        for name, param in encoder.named_parameters():
            print(name,param.shape)
            if "weight" in name:
                encoder_kernel_names.append(name)
                encoder_kernel_shapes.append(param.shape)
                logger.add_histogram("encoder_weight_"+name, param.clone().cpu().data.numpy(),0)

            if "bias" in name:
                encoder_bias_names.append(name)
                encoder_bias_shapes.append(param.shape)
                logger.add_histogram("encoder_bias_"+name, param.clone().cpu().data.numpy(),0)
            # if "final" in name:
            #     st()
            # summ_writer.summ_histogram(name, param.clone().cpu().data.numpy())
        
        for name, param in decoder.named_parameters():
            print(name,param.shape)
            if "weight" in name:
                decoder_kernel_names.append(name)
                decoder_kernel_shapes.append(param.shape)
                logger.add_histogram("decoder_weight_"+name, param.clone().cpu().data.numpy(),0)
            if "bias" in name:
                decoder_bias_names.append(name)
                decoder_bias_shapes.append(param.shape)
                logger.add_histogram("decoder_bias_"+name, param.clone().cpu().data.numpy(),0)
            # if "final" in name:
            #     st()
            # summ_writer.summ_histogram(name, param.clone().cpu().data.numpy())
        st()
        pickle.dump({"encoder_kernel":[encoder_kernel_names,encoder_kernel_shapes], "encoder_bias":[encoder_bias_names,encoder_bias_shapes], 
            "decoder_kernel":[decoder_kernel_names,decoder_kernel_shapes], "decoder_bias":[decoder_bias_names,decoder_bias_shapes]},open("hypernet.p","wb"))
        # st()
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    
    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']
        
        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)
        
        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {'query_crop_size': query_vol_size,
                         'input_crop_size': input_vol_size,
                         'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                         'reso': grid_reso}

    else: 
        vol_bound = None
        vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    input_type = cfg['data']['input_type']
    fields = {}
    # st()
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            if cfg['data']['dataloader_type'] == "normal":
                fields['points'] = data.PointsField(
                    cfg['data']['points_file'], points_transform,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files'],
                    cfg=cfg
                )
            else:
                fields['points'] = data.PointsField_Pydisco(
                    cfg['data']['points_file'], points_transform,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files'],
                    cfg=cfg
                )
        else:
            fields['points'] = data.PatchPointsField(
                cfg['data']['points_file'], 
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
                )
            else:
                if cfg['data']['dataloader_type'] == "normal":
                    fields['points_iou'] = data.PointsField(
                        points_iou_file,
                        unpackbits=cfg['data']['points_unpackbits'],
                        multi_files=cfg['data']['multi_files'],
                        cfg=cfg
                    )
                else:
                    fields['points_iou'] = data.PointsField_Pydisco(
                        points_iou_file,
                        unpackbits=cfg['data']['points_unpackbits'],
                        multi_files=cfg['data']['multi_files'],
                        cfg=cfg
                    )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)
    return fields
