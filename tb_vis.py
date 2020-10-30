import torch
import numpy as np
import ipdb 
import cv2
from itertools import combinations
import matplotlib
from src.common import normalize_coord_pydisco, coordinate2index
st = ipdb.set_trace
log_every = 1

def back2color(i):
    return ((i+0.5)*255).type(torch.ByteTensor)

def summ_rgb(name, logger, rgb, iteration):
    # if iteration%log_every:
    rgb = back2color(rgb)
    logger.add_image(name, rgb[0])

def summ_depth(name, logger, pix_T_camX, xyz_camX, H, W, it, return_only=False):
    depth_camX, _ = create_depth_image(pix_T_camX, xyz_camX, H, W)
    img = torch.zeros_like(depth_camX) - 0.5
    img[depth_camX<100] = 0.5
    img = img[0]
    # st()
    img = torch.cat([img, img, img], dim=0)
    if return_only:
        return img
    summ_rgb(f"Depth_{name}", logger, img.unsqueeze(0), it)

def summ_box(name, logger, rgb, bbox, pix_T_camX, it):
    # st()
    B, C, H, W = list(rgb.shape)
    boxes = get_alignedboxes2thetaformat(bbox.unsqueeze(0))
    box_corners = transform_boxes_to_corners_single(boxes[0])
    summ_box_by_corners(logger, name, it, rgb, box_corners.unsqueeze(1), torch.ones((1,1)).cuda(), torch.ones((1,1)).cuda(), pix_T_camX, only_return=False)

def summ_sdf_occupancies_single(name, logger, points_camXs, occs, rgb_camXs, pix_T_cams, it):
    rgb_camXs = rgb_camXs[0]
    pix_T_cams = pix_T_cams[0]
    occs = occs[0]
    points_camXs = points_camXs[0]

    _, H, W = rgb_camXs.shape

    occupied_idxs = torch.where(occs > 0.5)[0]
    free_idxs = torch.where(occs < 0.5)[0]
    points_proj = apply_pix_T_cam(pix_T_cams.unsqueeze(0), points_camXs.unsqueeze(0))
    points_proj = points_proj[0].long()

    points_proj[:,0] = torch.clamp(points_proj[:,0], 0, W-1)
    points_proj[:,1] = torch.clamp(points_proj[:,1], 0, H-1)

    occupied_points_proj = points_proj[occupied_idxs]
    free_points_proj = points_proj[free_idxs]

    occupied_mask = rgb_camXs.clone()
    occupied_mask[:, occupied_points_proj[:, 1], occupied_points_proj[:, 0]] = 0.5

    free_mask = rgb_camXs.clone()
    free_mask[:, free_points_proj[:, 1], free_points_proj[:, 0]] = 0.5
    # st()
    vis = torch.cat([occupied_mask, free_mask], dim=-1)
    summ_rgb(name, logger, vis.unsqueeze(0), it)
    return occupied_mask, free_mask

def summ_occ_grid(name, logger, bbox_ends, pix_T_camX, xyz_camX, res, it):
    # xyz_camX = xyz_camX[0]
    pix_T_camX = pix_T_camX[0]
    normalized_xyz = normalize_coord_pydisco(xyz_camX, bbox_ends[0], plane="grid")
    memcoord_xyz = coordinate2index(normalized_xyz, res, coord_type='3d')
    grid = torch.zeros(res*res*res).cuda()
    grid[memcoord_xyz[0,0]] = 1
    grid = grid.reshape(1, 1, res, res, res)

    # visualize along y axis
    # st()
    grid_reduce_z = convert_occ_to_height(grid, reduce_axis=2)
    grid_reduce_y = convert_occ_to_height(grid, reduce_axis=3)
    grid_reduce_x = convert_occ_to_height(grid, reduce_axis=4)
    grid_vis = torch.cat([grid_reduce_z, grid_reduce_y, grid_reduce_x], dim=-1)
    grid_vis = grid_vis > 0.5
    grid_vis = grid_vis.int() - 0.5
    grid_vis = grid_vis.repeat(1,3,1,1)
    summ_rgb(name, logger, grid_vis, it)
    

def convert_occ_to_height(occ, reduce_axis=3):
	B, C, D, H, W = list(occ.shape)
	assert(C==1)
	# note that height increases DOWNWARD in the tensor
	# (like pixel/camera coordinates)
	
	G = list(occ.shape)[reduce_axis]
	values = torch.linspace(float(G), 1.0, steps=G).type(torch.FloatTensor).cuda()
	if reduce_axis==2:
		# frontal view
		values = values.view(1, 1, G, 1, 1)
	elif reduce_axis==3:
		# top view
		values = values.view(1, 1, 1, G, 1)
	elif reduce_axis==4:
		# lateral view
		values = values.view(1, 1, 1, 1, G)
	else:
		assert(False) # you have to reduce one of the spatial dims (2-4)
	values = torch.max(occ*values, dim=reduce_axis)[0]/float(G)
	# values = values.view([B, C, D, W])
	return values

def summ_box_by_corners(logger, name, it, rgbR, corners, scores, tids, pix_T_cam, only_return=False):
    # rgb is B x H x W x C
    # corners is B x N x 8 x 3 
    # scores is B x N
    # tids is B x N
    # pix_T_cam is B x 4 x 4
    # st()
    B, C, H, W = list(rgbR.shape)
    boxes_vis = draw_corners_on_image(rgbR,
                                            corners,
                                            scores,
                                            tids,
                                            pix_T_cam,None)
    if not only_return:
        summ_rgb(name, logger, boxes_vis, it)
    return boxes_vis


def draw_corners_on_image(rgb, corners_cam, scores, tids, pix_T_cam,info_text=None):
    # first we need to get rid of invalid gt boxes
    # gt_boxes = trim_gt_boxes(gt_boxes)
    B, C, H, W = list(rgb.shape)
    assert(C==3)
    B2, N, D, E = list(corners_cam.shape)
    assert(B2==B)
    assert(D==8) # 8 corners
    assert(E==3) # 3D

    rgb = back2color(rgb)
    corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
    corners_pix_ = apply_pix_T_cam(pix_T_cam, corners_cam_)
    corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
    out = draw_boxes_on_image_py(rgb[0].cpu().numpy(),
                                        corners_pix[0].cpu().numpy(),
                                        scores[0].cpu().numpy(),
                                        tids[0].cpu().numpy(),info_text)
    out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
    out = torch.unsqueeze(out, dim=0)
    out = preprocess_color(out)
    out = torch.reshape(out, [1, C, H, W])
    return out

def draw_boxes_on_image_py(rgb, corners_pix, scores, tids,info_text=None, boxes=None, thickness=1,text=False):
    # all inputs are numpy tensors
    # rgb is H x W x 3
    # corners_pix is N x 8 x 2, in xy order
    # scores is N
    # tids is N
    # boxes is N x 9 < this is only here to print some rotation info
    # pix_T_cam is 4 x 4
    rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    H, W, C = rgb.shape
    assert(C==3)
    N, D, E = corners_pix.shape
    assert(D==8)
    assert(E==2)

    if boxes is not None:
        rx = boxes[:,6]
        ry = boxes[:,7]
        rz = boxes[:,8]
    else:
        rx = 0
        ry = 0
        rz = 0

    color_map = matplotlib.cm.get_cmap('tab20')
    color_map = color_map.colors

    # draw
    for ind, corners in enumerate(corners_pix):
        # corners is 8 x 2
        # st()
        if not np.isclose(scores[ind], 0.0):
            # print 'score = %.2f' % scores[ind]
            color_id = tids[ind] % 20
            color = color_map[2]
            color_text = color_map[2]

            # st()

            color = np.array(color)*255.0
            # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])
            if info_text is not None:
                text_to_put = info_text[ind]
                cv2.putText(rgb,
                            text_to_put, 
                            (np.min(corners[:,0]), np.min(corners[:,1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, # font size
                            color_text,
                            2) # font weight

            for c in corners:

                # rgb[pt1[0], pt1[1], :] = 255
                # rgb[pt2[0], pt2[1], :] = 255
                # rgb[np.clip(int(c[0]), 0, W), int(c[1]), :] = 255

                c0 = np.clip(int(c[0]), 0,  W-1)
                c1 = np.clip(int(c[1]), 0,  H-1)
                rgb[c1, c0, :] = 255

            # we want to distinguish between in-plane edges and out-of-plane ones
            # so let's recall how the corners are ordered:
            xs = np.array([-1/2., -1/2., -1/2., -1/2., 1/2., 1/2., 1/2., 1/2.])
            ys = np.array([-1/2., -1/2., 1/2., 1/2., -1/2., -1/2., 1/2., 1/2.])
            zs = np.array([-1/2., 1/2., -1/2., 1/2., -1/2., 1/2., -1/2., 1/2.])
            xs = np.reshape(xs, [8, 1])
            ys = np.reshape(ys, [8, 1])
            zs = np.reshape(zs, [8, 1])
            offsets = np.concatenate([xs, ys, zs], axis=1)

            corner_inds = list(range(8))
            combos = list(combinations(corner_inds, 2))

            for combo in combos:
                pt1 = offsets[combo[0]]
                pt2 = offsets[combo[1]]
                # draw this if it is an in-plane edge
                eqs = pt1==pt2
                if np.sum(eqs)==2:
                    i, j = combo
                    pt1 = (corners[i, 0], corners[i, 1])
                    pt2 = (corners[j, 0], corners[j, 1])
                    retval, pt1, pt2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
                    if retval:
                        cv2.line(rgb, pt1, pt2, color, thickness, cv2.LINE_AA)

                    # rgb[pt1[0], pt1[1], :] = 255
                    # rgb[pt2[0], pt2[1], :] = 255
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # utils_basic.print_stats_py('rgb_uint8', rgb)
    # imageio.imwrite('boxes_rgb.png', rgb)
    return rgb

def preprocess_color(x):
	if type(x).__module__ == np.__name__:
		return x.astype(np.float32) * 1./255 - 0.5
	else:
		return x.float() * 1./255 - 0.5

def summ_pointcloud(logger, xyz, iteration):
    if iteration%log_every:
        logger.add_mesh("Pointcloud", vertices=xyz, global_step=iteration)

def get_alignedboxes2thetaformat(aligned_boxes):
    B,N,_,_ = list(aligned_boxes.shape)
    aligned_boxes = torch.reshape(aligned_boxes,[B,N,6])
    B,N,_ = list(aligned_boxes.shape)
    xmin,ymin,zmin,xmax,ymax,zmax = torch.unbind(torch.tensor(aligned_boxes), dim=-1)
    xc = (xmin+xmax)/2.0
    yc = (ymin+ymax)/2.0
    zc = (zmin+zmax)/2.0
    w = xmax-xmin
    h = ymax - ymin
    d = zmax - zmin
    zeros = torch.zeros([B,N]).cuda()
    boxes = torch.stack([xc,yc,zc,w,h,d,zeros,zeros,zeros],dim=-1)
    return boxes

def transform_boxes_to_corners_single(boxes):
    N, D = list(boxes.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = torch.unbind(boxes, axis=1)
    # these are each shaped N

    ref_T_obj = convert_box_to_ref_T_obj(boxes)

    xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
    ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
    zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
    
    xyz_obj = torch.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref
    
    
def convert_box_to_ref_T_obj(box3D):
    # turn the box into obj_T_ref (i.e., obj_T_cam)
    B = list(box3D.shape)[0]
    
    # box3D is B x 9
    x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box3D, axis=1)
    rot0 = eye_3x3(B)
    tra = torch.stack([x, y, z], axis=1)
    center_T_ref = merge_rt(rot0, -tra)
    # center_T_ref is B x 4 x 4
    
    t0 = torch.zeros([B, 3])
    rot = eul2rotm(rx, -ry, -rz)
    obj_T_center = merge_rt(rot, t0)
    # this is B x 4 x 4

    # we want obj_T_ref
    # first we to translate to center,
    # and then rotate around the origin
    obj_T_ref = matmul2(obj_T_center, center_T_ref)

    # return the inverse of this, so that we can transform obj corners into cam coords
    ref_T_obj = obj_T_ref.inverse()
    return ref_T_obj

def eul2rotm(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = torch.unsqueeze(rx, dim=1)
    ry = torch.unsqueeze(ry, dim=1)
    rz = torch.unsqueeze(rz, dim=1)
    # these are B x 1
    sinz = torch.sin(rz)
    siny = torch.sin(ry)
    sinx = torch.sin(rx)
    cosz = torch.cos(rz)
    cosy = torch.cos(ry)
    cosx = torch.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=2)
    r2 = torch.stack([r21,r22,r23],dim=2)
    r3 = torch.stack([r31,r32,r33],dim=2)
    r = torch.cat([r1,r2,r3],dim=1)
    return r

def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

def eye_4x4(B):
    rt = torch.eye(4, device=torch.device('cuda')).view(1,4,4).repeat([B, 1, 1])
    return rt

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def eye_3x3(B):
    rt = torch.eye(3, device=torch.device('cuda')).view(1,3,3).repeat([B, 1, 1])
    return rt

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


def create_depth_image_single(xy, z, H, W):
    # turn the xy coordinates into image inds
    #print(hashit(xy),hashit(z))
    xy = torch.round(xy).long()
    #print(hashit(xy))
    depth = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
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

    #print(hashit(inds))
    depth[inds] = z
    #print(hashit(depth))
    # st()
    valid = (depth > 0.0).float()
    # print(torch.sum(depth))
    depth[torch.where(depth == 0.0)] = 100.0
    # print(torch.sum(depth))
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def create_depth_image(pix_T_cam, xyz_cam, H, W):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]

    depth = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    for b in range(B):
        depth[b], valid[b] = create_depth_image_single(xy[b], z[b], H, W)
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


def sub2ind(height, width, y, x):
    return y*width + x