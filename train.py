import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import ipdb
import getpass
st = ipdb.set_trace
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
import tb_vis
import ipdb 
st = ipdb.set_trace


# debug = True




# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--run_name', type=str, help='runname', default='run1')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--debug', action='store_true', help='debug mode.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('--log_every', type=int, default=500,
                    help='Number of iterations to log after')
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")



username = getpass.getuser()

if args.debug:
    num_val = 5
    args.log_every = 1
else:
    num_val = 500
# Set t0
import socket
if "compute" in socket.gethostname():
    if cfg['data']['dataloader_type'] == 'pydisco':
        num_val = 185
        if username == "shamitl":
            cfg['data']['path'] ='/home/shamitl/datasets/shapenet_renders/npys'
        else:
            cfg['data']['path'] ='/home//mprabhud/dataset/shapenet_renders/npys'
    else:
        cfg['data']['path'] = "/projects/katefgroup/datasets/ShapeNet/"
else:
    if cfg['data']['dataloader_type'] == 'pydisco':
        cfg['data']['path'] = "/media/mihir/dataset/shapenet_renders/npys"
    else:
        cfg['data']['path'] = "data/ShapeNet"

t0 = time.time()
import os


exp_name = args.config.split("/")[-1][:-5]

# st()

cfg['training']['out_dir'] = os.path.join('out/pointcloud',exp_name)

if cfg['training']['load_exp'] != "nothing":
    load_dir = os.path.join('out/pointcloud',cfg['training']['load_exp'])
else:
    load_dir = os.path.join('out/pointcloud',exp_name)

# st()
# Shorthands

out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))



# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg, return_idx=True)
# st()
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, num_workers=0, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=0, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

model_counter = defaultdict(int)
data_vis_list = []

# Build a data dictionary for visualization
iterator = iter(vis_loader)
for i in range(num_val):
    print(i)
    data_vis = next(iterator)
    idx = data_vis['idx'].item()
    model_dict = val_dataset.get_model_dict(idx)
    category_id = model_dict.get('category', 'n/a')
    category_name = val_dataset.metadata[category_id].get('name', 'n/a')
    category_name = category_name.split(',')[0]
    if category_name == 'n/a':
        category_name = category_id

    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

    model_counter[category_id] += 1
# st()
# Model
logger = SummaryWriter(os.path.join(out_dir, 'logs', args.run_name))

model = config.get_model(cfg, device=device, dataset=train_dataset, logger=logger)

lr = cfg['training']['lr']
# Generator
generator = config.get_generator(model, cfg, device=device)

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

# st()
checkpoint_io = CheckpointIO(load_dir, model=model, optimizer=optimizer)

load_dict = dict()


if not cfg['training']['no_load']:
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()

epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))



# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']
dynamic_dict = cfg['model']['hypernet_params']['dynamic_dict']
vq_loss_coeff = cfg['model']['hypernet_params']['vq_loss_coeff']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)
print('output path: ', cfg['training']['out_dir'])

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1

        bbox_ends = batch['inputs.bbox_ends']

        if it % args.log_every == 0:
            # Visualize stuff
            if 'inputs.single_view_rgb' in batch:
                tb_vis.summ_rgb("rgb_camX", logger, batch['inputs.single_view_rgb'], it)
                tb_vis.summ_box("bbox", logger, batch['inputs.single_view_rgb'], bbox_ends.cuda(), batch['inputs.pix_T_camX'].cuda(), it)
                tb_vis.summ_sdf_occupancies_single("SDF_sampled", logger, batch['points'].cuda(), batch['points.occ'].cuda(), batch['inputs.single_view_rgb'].cuda(), batch['inputs.pix_T_camX'].cuda(), it)
                tb_vis.summ_sdf_occupancies_single("SDF_all", logger, batch['points.points_all'].cuda(), batch['points.occupancies_all'].cuda(), batch['inputs.single_view_rgb'].cuda(), batch['inputs.pix_T_camX'].cuda(), it)

            # tb_vis.summ_depth("Depth_all", logger, batch['inputs.pix_T_camX'].cuda(), batch['inputs.points_all'].cuda(), cfg['data']['height'], cfg['data']['width'], it)
            tb_vis.summ_depth("Depth_sampled", logger, batch['inputs.pix_T_camX'].cuda(), batch['inputs'].cuda(), cfg['data']['height'], cfg['data']['width'], it)
            
            # tb_vis.summ_occ_grid("Occ_all", logger, bbox_ends.cuda(), batch['inputs.pix_T_camX'].cuda(), batch['inputs.points_all'].cuda(), cfg['model']['encoder_kwargs']['grid_resolution'], it)
            tb_vis.summ_occ_grid("Occ_sampled", logger, bbox_ends.cuda(), batch['inputs.pix_T_camX'].cuda(), batch['inputs'].cuda(), cfg['model']['encoder_kwargs']['grid_resolution'], it)

        print("Processing iteration: ", it)
        # st()
        from scipy.misc import imsave
        # st()
        # imsave("out.png",batch['inputs.single_view_rgb'][0].permute(1,2,0).cpu().numpy())
        # st()
        if cfg['data']['single_view_pcd']:
            rgb = batch['inputs.single_view_rgb'].to(device)
        else:
            rgb = None

        arg_dict = {'logger':logger, 'iteration':it, 'dynamic_dict':dynamic_dict, 'rgb':rgb}

        loss = trainer.train_step(batch,vq_loss_coeff=vq_loss_coeff,arg_dict=arg_dict)


        logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

        # Visualize output
        if (visualize_every > 0 and (it % visualize_every) == 0) and False:
            st()
            print('Visualizing')
            for data_vis in data_vis_list:
                if cfg['generation']['sliding_window']:
                    out = generator.generate_mesh_sliding(data_vis['data'])    
                else:
                    out = generator.generate_mesh(data_vis['data'])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}

                mesh.export(os.path.join(out_dir, 'vis', '{}_{}_{}.off'.format(it, data_vis['category'], data_vis['it'])))
                st()

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0) and False:
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        # st()
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
