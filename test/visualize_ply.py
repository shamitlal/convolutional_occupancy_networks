import numpy as np
import open3d as o3d 
import pickle
import torch
import ipdb 
st = ipdb.set_trace

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

def make_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    return pcd

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

pcd_list = [mesh_frame]
# for i in range(10):
#     path = f"/Users/shamitlal/Desktop/temp/convocc/pointcloud_0{i}.npz"
#     pcd = np.load(path)['points']
#     st()
#     pcd = make_pcd(pcd)
#     pcd_list.append(pcd)

path = f"/Users/shamitlal/Desktop/temp/convocc/pointcloud.npz"
pcd = np.load(path)['points']
print("max: ", np.max(pcd, axis=0))
print("min: ", np.min(pcd, axis=0))
pcd = make_pcd(pcd)
pcd_list.append(pcd)
print("Pcd list len is: ", len(pcd_list))
o3d.visualization.draw_geometries(pcd_list)

# Visualize inside points
# path = f"/Users/shamitlal/Desktop/temp/convocc/points.npz"
# pcd = np.load(path)
# occ = np.unpackbits(pcd['occupancies'])
# pcd = pcd['points']
# occ_pts_idx = np.where(occ==1)[0]
# pcd = pcd[occ_pts_idx]
# print("max: ", np.max(pcd, axis=0))
# print("min: ", np.min(pcd, axis=0))
# pcd = make_pcd(pcd)
# pcd_list.append(pcd)


# # Visualize actual pointcloud
# path = f"/Users/shamitlal/Desktop/temp/convocc/pointcloud.npz"
# pcd = np.load(path)['points']
# print("max: ", np.max(pcd, axis=0))
# print("min: ", np.min(pcd, axis=0))
# pcd = make_pcd(pcd)
# pcd_list.append(pcd)

# print("Pcd list len is: ", len(pcd_list))
o3d.visualization.draw_geometries(pcd_list)

#Visualize pydisco shapenet data
# path = f"/Users/shamitlal/Desktop/temp/convocc/02958343_c48a804986a819b4bda733a39f84326d.p"
# pfile = pickle.load(open(path, "rb"))
# xyz_camXs = torch.tensor(pfile['xyz_camXs_raw'])
# origin_T_camXs = torch.tensor(pfile['origin_T_camXs_raw'])
# xyz_origin = apply_4x4(origin_T_camXs, xyz_camXs)
# pcd = xyz_origin.reshape(-1, 3)
# x, y, z = torch.abs(pcd[:,0]), torch.abs(pcd[:,1]), torch.abs(pcd[:,2])
# cond1 = (x<10)
# cond2 = (y<10)
# cond3 = (z<10) 
# cond = cond1 & cond2 & cond3
# pcd = pcd[cond]
# pcd_list.append(make_pcd(pcd))
# o3d.visualization.draw_geometries(pcd_list)
# st()
# aa=1