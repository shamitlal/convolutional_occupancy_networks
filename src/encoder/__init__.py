from src.encoder import (
    pointnet, voxels, pointnetpp
)


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_local_pool_hyper': pointnet.LocalPoolPointnet_Hyper,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus_ssg': pointnetpp.PointNetPlusPlusSSG,
    'pointnet_plus_plus_msg': pointnetpp.PointNetPlusPlusMSG,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
}
