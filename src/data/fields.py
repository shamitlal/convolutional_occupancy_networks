import os
import glob
import random
import ipdb
st = ipdb.set_trace
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field
from src.utils import binvox_rw
from src.common import coord2index, normalize_coord
from torch_scatter import scatter
import pickle
import ipdb
st = ipdb.set_trace
import torch
import imageio
class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category, camera_view):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

# 3D Fields
class PatchPointsField(Field):
    ''' Patch Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape and then split to patches.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        
    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # acquire the crop
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['query_vol'][0][i])
                     & (points[:, i] <= vol['query_vol'][1][i]))
        ind = ind_list[0] & ind_list[1] & ind_list[2]
        data = {None: points[ind],
                    'occ': occupancies[ind],
            }
            
        if self.transform is not None:
            data = self.transform(data)

        # calculate normalized coordinate w.r.t. defined query volume
        p_n = {}
        for key in vol['plane_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(data[None].copy(), vol['input_vol'], plane=key)
        data['normalized'] = p_n

        return data

class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None, cfg=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        self.cfg = cfg

    def load(self, model_path, idx, category, camera_view):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        # st()
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        # st()
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]

        occupancies = occupancies.astype(np.float32)

        if self.cfg['data']['warp_to_camera_frame'] or self.cfg['data']['single_view_pcd']:
            camera_path = os.path.join(model_path, 'img_choy2016', 'cameras.npz')
            camera = np.load(camera_path)
            camXV_T_origin = torch.tensor(get_4x4(camera[f'world_mat_{camera_view}'])).unsqueeze(0)
            points = apply_4x4(camXV_T_origin, torch.tensor(points).unsqueeze(0))
            points = points.squeeze(0).numpy()
        data = {
            None: points,
            'occ': occupancies,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class PointsField_Pydisco(Field):
    ''' Point Field Pydisco.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None, cfg=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        self.cfg = cfg

    def get_bounding_box(self, points):

        points = points[0]
        xmin, ymin, zmin = torch.min(points, dim=0)[0]
        xmax, ymax, zmax = torch.max(points, dim=0)[0]
        
        return torch.tensor([[xmin-0.15, ymin-0.15, zmin-0.15],[xmax+0.15, ymax+0.15, zmax+0.15]])

    def load(self, model_path, idx, category, camera_view):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = pickle.load(open(model_path, "rb"))


        xyz_camX = points_dict['xyz_camXs_raw']
        origin_T_camXs = points_dict['origin_T_camXs_raw']
        xyz_origin = apply_4x4(torch.tensor(origin_T_camXs), torch.tensor(xyz_camX))
        pcd = xyz_origin.reshape(-1, 3)
        x, y, z = torch.abs(pcd[:,0]), torch.abs(pcd[:,1]), torch.abs(pcd[:,2])
        cond1 = (x<20)
        cond2 = (y<20)
        cond3 = (z<20) 
        cond = cond1 & cond2 & cond3
        xyz_origin = pcd[cond]
        bbox_ends = self.get_bounding_box(torch.tensor(xyz_origin).unsqueeze(0)) # bbox in origin        


        points = points_dict['sdf_points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['sdf_occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        if self.cfg['data']['warp_to_camera_frame'] or self.cfg['data']['single_view_pcd']:
            origin_T_camXV = points_dict['origin_T_camXs_raw'][camera_view]
            camXV_T_origin = safe_inverse(torch.tensor(origin_T_camXV).unsqueeze(0))
            points = apply_4x4(camXV_T_origin, torch.tensor(points).unsqueeze(0))
            points = points.squeeze(0).numpy()

            xyz_camXV = apply_4x4(camXV_T_origin, torch.tensor(xyz_origin).unsqueeze(0))
            xyz_camXV = xyz_camXV.numpy()[0]
            bbox_ends = self.get_bounding_box(torch.tensor(xyz_camXV).unsqueeze(0))

        # Only take sdfs inside bbox
        x, y, z = points[:, 0],points[:, 1],points[:, 2]
        cond1 = x > bbox_ends.numpy()[0,0]
        cond2 = x < bbox_ends.numpy()[1,0]
        cond3 = y > bbox_ends.numpy()[0,1]
        cond4 = y < bbox_ends.numpy()[1,1]
        cond5 = z > bbox_ends.numpy()[0,2]
        cond6 = z < bbox_ends.numpy()[1,2]
        cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 

        points = points[cond]
        occupancies = occupancies[cond]
        # st()
        num = points.shape[0]
        all_indexes = list(range(num))
        random.seed(1)
        random.shuffle(all_indexes)
        filtered_indexes = all_indexes[:100000]

        points = points[filtered_indexes]
        occupancies = occupancies[filtered_indexes]

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.transform is not None:
            data = self.transform(data)
        
        if points.shape[0]<100000:
            num_repeat = 100000//points.shape[0] +   1
            points = np.tile(np.expand_dims(points, 0), (num_repeat, 1, 1)).reshape(-1, 3)[:100000]
            occupancies = np.tile(np.expand_dims(occupancies, 0), (num_repeat, 1)).reshape(-1)[:100000]

        data['points_all'] = points
        data['occupancies_all'] = occupancies
        return data



def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PatchPointCloudField(Field):
    ''' Patch point cloud field.

    It provides the field used for patched point cloud data. These are the points
    randomly sampled on the mesh and then partitioned.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, transform_add_noise=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        # st()
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        # add noise globally
        if self.transform is not None:
            data = {None: points, 
                    'normals': normals}
            data = self.transform(data)
            points = data[None]

        # acquire the crop index
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['input_vol'][0][i])
                    & (points[:, i] <= vol['input_vol'][1][i]))
        mask = ind_list[0] & ind_list[1] & ind_list[2]# points inside the input volume
        mask = ~mask # True means outside the boundary!!
        data['mask'] = mask
        points[mask] = 0.0
        
        # calculate index of each point w.r.t. defined resolution
        index = {}
        
        for key in vol['plane_type']:
            index[key] = coord2index(points.copy(), vol['input_vol'], reso=vol['reso'], plane=key)
            if key == 'grid':
                index[key][:, mask] = vol['reso']**3
            else:
                index[key][:, mask] = vol['reso']**2
        data['ind'] = index
        
        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, multi_files=None, cfg=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.cfg=cfg

    def get_bounding_box(self, points):

        points = points[0]
        xmin, ymin, zmin = torch.min(points, dim=0)[0]
        xmax, ymax, zmax = torch.max(points, dim=0)[0]
        
        return torch.tensor([[xmin-0.15, ymin-0.15, zmin-0.15],[xmax+0.15, ymax+0.15, zmax+0.15]])

    def load(self, model_path, idx, category, camera_view):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        bbox_ends = self.get_bounding_box(torch.tensor(points).unsqueeze(0))
        camXV_T_origin = None
        pix_T_camX = None
        # st()
        if self.cfg['data']['warp_to_camera_frame'] or self.cfg['data']['single_view_pcd']:
            camera_path = os.path.join(model_path, 'img_choy2016', 'cameras.npz')
            camera = np.load(camera_path)
            camXV_T_origin = torch.tensor(get_4x4(camera[f'world_mat_{camera_view}'])).unsqueeze(0)
            pix_T_camX = torch.tensor(camera[f'camera_mat_{camera_view}']).unsqueeze(0).float()

            points = apply_4x4(camXV_T_origin, torch.tensor(points).unsqueeze(0))
            bbox_ends = self.get_bounding_box(points)

            depth, _ = create_depth_image_from_complete_pointcloud(pix_T_camX, points, self.cfg['data']['height'], self.cfg['data']['width'])
            xyz_camX = depth2pointcloud_cpu(depth, pix_T_camX)
            xyz_camX = xyz_camX[0]
            zs = xyz_camX[:,2]
            valid_mask = zs < 100
            xyz_camX = xyz_camX[valid_mask]
            # st()
            num_request = torch.ceil(torch.tensor(self.cfg['data']['pointcloud_n']/xyz_camX.shape[0]))
            num_request = int(num_request.item())
            if num_request > 1:
                xyz_camX = xyz_camX.unsqueeze(0).repeat(num_request,1,1).reshape(-1,3)
            points = points.numpy()[0]
            if self.cfg['data']['single_view_pcd']:
                points=xyz_camX.numpy()
                rgb = imageio.imread(os.path.join(model_path, 'img_choy2016', str(camera_view).zfill(3)+".jpg"))
                rgb = torch.tensor(rgb)
                if rgb.ndim == 2:
                    rgb = rgb.unsqueeze(-1).repeat(1,1,3)
                rgb = rgb.permute(2,0,1)/255. - 0.5

        data = {
            None: points,
            'normals': normals,
        }

        if self.transform is not None:
            data = self.transform(data)
        # st()
        data['bbox_ends'] =  bbox_ends #torch.tensor([[-0.5,-0.5,-0.5],[0.5,0.5,0.5]])  

        if self.cfg['data']['warp_to_camera_frame']:
            data['pix_T_camX'] = pix_T_camX.squeeze(0)
            data['camX_T_origin'] = camXV_T_origin.squeeze(0)
        if self.cfg['data']['single_view_pcd']:
            data['single_view_rgb'] = rgb.numpy()

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete



class PointCloudField_Pydisco(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, multi_files=None, cfg=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.cfg=cfg

    def get_bounding_box(self, points):

        points = points[0]
        xmin, ymin, zmin = torch.min(points, dim=0)[0]
        xmax, ymax, zmax = torch.max(points, dim=0)[0]
        
        return torch.tensor([[xmin - 0.15, ymin - 0.15, zmin - 0.15],[xmax + 0.15, ymax + 0.15, zmax + 0.15]])

    def load(self, model_path, idx, category, camera_view):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        # st()
        # st()
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        # pointcloud_dict = np.load(file_path)
        pointcloud_dict = pickle.load(open(model_path, "rb"))

        points_camX = pointcloud_dict['xyz_camXs_raw']
        origin_T_camXs = pointcloud_dict['origin_T_camXs_raw']
        camXV_T_origin = safe_inverse(torch.tensor(origin_T_camXs))
        pix_T_camX = pointcloud_dict['pix_T_cams_raw'][camera_view]

        points = apply_4x4(torch.tensor(origin_T_camXs), torch.tensor(points_camX))
        pcd = points.reshape(-1, 3)
        x, y, z = torch.abs(pcd[:,0]), torch.abs(pcd[:,1]), torch.abs(pcd[:,2])
        cond1 = (x<20)
        cond2 = (y<20)
        cond3 = (z<20) 
        cond = cond1 & cond2 & cond3
        points = pcd[cond]
        # points = pointcloud_dict['points'].astype(np.float32)
        # normals = pointcloud_dict['normals'].astype(np.float32)
        bbox_ends = self.get_bounding_box(torch.tensor(points).unsqueeze(0))
        if self.cfg['data']['warp_to_camera_frame'] or self.cfg['data']['single_view_pcd']:
            # st()
            camXV_T_origin = camXV_T_origin[camera_view]
            points = apply_4x4(camXV_T_origin.unsqueeze(0), torch.tensor(points).unsqueeze(0))
            points = points.numpy()[0]
            bbox_ends = self.get_bounding_box(torch.tensor(points).unsqueeze(0))

            rgb = pointcloud_dict['rgb_camXs_raw'][camera_view]
            rgb = torch.tensor(rgb)
            if rgb.ndim == 2:
                rgb = rgb.unsqueeze(-1).repeat(1,1,3)
            rgb = rgb.permute(2,0,1)/255. - 0.5
           
            if self.cfg['data']['single_view_pcd']:
                pcd = torch.tensor(points_camX[camera_view])
                x, y, z = torch.abs(pcd[:,0]), torch.abs(pcd[:,1]), torch.abs(pcd[:,2])
                cond1 = (x<20)
                cond2 = (y<20)
                cond3 = (z<20) 
                cond = cond1 & cond2 & cond3
                points = pcd[cond]

                num_request = torch.ceil(torch.tensor(self.cfg['data']['pointcloud_n']/points.shape[0]))
                num_request = int(num_request.item())
                if num_request > 1:
                    points = points.unsqueeze(0).repeat(num_request,1,1).reshape(-1,3)

                points=points.numpy()

        points_all = points
        data = {
            None: points,
            'normals': points, # just dummy value, i think its used only in qualitative mesh eval
        }
        # st()
        if self.transform is not None:
            data = self.transform(data)
        
        data['bbox_ends'] = bbox_ends
        data['pix_T_camX'] = pix_T_camX
        data['camX_T_origin'] = camXV_T_origin
        # data['points_all'] = points_all
        if self.cfg['data']['warp_to_camera_frame'] or self.cfg['data']['single_view_pcd']:
            data['single_view_rgb'] = rgb.numpy()
        # st()
        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PartialPointCloudField(Field):
    ''' Partial Point cloud field.

    It provides the field used for partial point cloud data. These are the points
    randomly sampled on the mesh and a bounding box with random size is applied.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
        part_ratio (float): max ratio for the remaining part
    '''
    def __init__(self, file_name, transform=None, multi_files=None, part_ratio=0.7):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.part_ratio = part_ratio

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        
        side = np.random.randint(3)
        xb = [points[:, side].min(), points[:, side].max()]
        length = np.random.uniform(self.part_ratio*(xb[1] - xb[0]), (xb[1] - xb[0]))
        ind = (points[:, side]-xb[0])<= length
        data = {
            None: points[ind],
            'normals': normals[ind],
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


def get_4x4(mat):
    mat = torch.tensor(mat)
    new_mat = torch.zeros((4,4))
    new_mat[3,3] = 1
    new_mat[:3,:4] = mat
    return new_mat.float()

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2



def create_depth_image_from_complete_pointcloud_single(xy, z, H, W):
    # turn the xy coordinates into image inds
    #print(hashit(xy),hashit(z))
    xy = torch.round(xy).long()
    #print(hashit(xy))
    depth = torch.zeros(H*W, dtype=torch.float32)
    #print(hashit(depth))
    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid = (xy[:,0] <= W-1) & (xy[:,1] <= H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (z[:] > 0)
    #print(hashit(valid))
    # st()

    # gather these up
    xy = xy[valid]
    z = z[valid]

    #print(hashit(xy),hashit(z))

    inds = sub2ind(H, W, xy[:,1], xy[:,0]).long()
    out = scatter(z, inds, reduce="min")

    depth[torch.arange(out.shape[0])] = out
    valid = (depth > 0.0).float()
    # print(torch.sum(depth))
    depth[torch.where(depth == 0.0)] = 100.0
    # print(torch.sum(depth))
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def sub2ind(height, width, y, x):
    return y*width + x

def create_depth_image_from_complete_pointcloud(pix_T_cam, xyz_cam, H, W):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]
    # st()
    depth = torch.zeros(B, 1, H, W, dtype=torch.float32)#, device=torch.device(device))
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32)#, device=torch.device(device))
    for b in range(B):
        depth[b], valid[b] = create_depth_image_from_complete_pointcloud_single(xy[b], z[b], H, W)
    return depth, valid


def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS=1e-6
    x = (x*fx)/(z+EPS)+x0
    y = (y*fy)/(z+EPS)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def depth2pointcloud_cpu(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    y, x = meshgrid2D_cpu(B, H, W)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def meshgrid2D_cpu(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X
    grid_y = torch.linspace(0.0, Y-1, Y)
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X)
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2D(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def normalize_grid2D(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    B, H, W = list(z.shape)

    fx = torch.reshape(fx, [B,1,1])
    fy = torch.reshape(fy, [B,1,1])
    x0 = torch.reshape(x0, [B,1,1])
    y0 = torch.reshape(y0, [B,1,1])
    
    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)
    
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    xyz = torch.stack([x,y,z], dim=2)
    return xyz

def make_bbox_cube(bbox_ends):

    # center = (bbox_ends[1]+bbox_ends[0])/2
    length = bbox_ends[1]-bbox_ends[0]
    max_len = length.max()
    extra_len = max_len - length
    extra_len_size = extra_len/2.

    bbox_ends_new = bbox_ends.clone()

    bbox_ends_new[0] = bbox_ends_new[0] - extra_len_size
    bbox_ends_new[1] = bbox_ends_new[1] + extra_len_size
    return bbox_ends_new