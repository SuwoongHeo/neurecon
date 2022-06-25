import pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import argparse
from glob import glob
import os
import colmap_read_model as read_model
import shutil
from PIL import Image
from utils.io_util import center_resize_info
def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def get_Ps(cameras,number_of_cameras):
    Ps = []
    for i in range(0, number_of_cameras):
        P = cameras['world_mat_%d' % i][:3, :].astype(np.float64)
        Ps.append(P)
    return np.array(Ps)


#Gets the fundamental matrix that transforms points from the image of camera 2, to a line in the image of
#camera 1
def get_fundamental_matrix(P_1,P_2):
    P_2_center=np.linalg.svd(P_2)[-1][-1, :]
    epipole=P_1@P_2_center
    epipole_cross=np.zeros((3,3))
    epipole_cross[0,1]=-epipole[2]
    epipole_cross[1, 0] = epipole[2]

    epipole_cross[0,2]=epipole[1]
    epipole_cross[2, 0] = -epipole[1]

    epipole_cross[1, 2] = -epipole[0]
    epipole_cross[2, 1] = epipole[0]

    F = epipole_cross@P_1 @ np.linalg.pinv(P_2)
    return F



# Given a point (curx,cury) in image 0, get the  maximum and minimum
# possible depth of the point, considering the second image silhouette (index j)
def get_min_max_d(curx, cury, P_j, silhouette_j, P_0, Fj0, j):
    # transfer point to line using the fundamental matrix:
    cur_l_1=Fj0 @ np.array([curx,cury,1.0]).astype(np.float)
    cur_l_1 = cur_l_1 / np.linalg.norm(cur_l_1[:2])

    # Distances of the silhouette points from the epipolar line:
    dists = np.abs(silhouette_j.T @ cur_l_1)
    relevant_matching_points_1 = silhouette_j[:, dists < 0.7]

    if relevant_matching_points_1.shape[1]==0:
        return (0.0,0.0)
    X = cv2.triangulatePoints(P_0, P_j, np.tile(np.array([curx, cury]).astype(np.float),
                                                (relevant_matching_points_1.shape[1], 1)).T,
                              relevant_matching_points_1[:2, :])
    depths = P_0[2] @ (X / X[3])
    reldepth=depths >= 0
    depths=depths[reldepth]
    if depths.shape[0] == 0:
        return (0.0, 0.0)

    min_depth = depths.min()
    max_depth = depths.max()

    return min_depth,max_depth

#get all fundamental matrices that trasform points from camera 0 to lines in Ps
def get_fundamental_matrices(P_0, Ps):
    Fs=[]
    for i in range(0,Ps.shape[0]):
        F_i0 = get_fundamental_matrix(Ps[i],P_0)
        Fs.append(F_i0)
    return np.array(Fs)

def get_all_mask_points(masks_dir):
    mask_paths = sorted(glob_imgs(masks_dir))
    mask_points_all=[]
    mask_ims = []
    for path in mask_paths:
        img = mpimg.imread(path)
        cur_mask = img.max(axis=2) > 0.5
        mask_points = np.where(img.max(axis=2) > 0.5)
        xs = mask_points[1]
        ys = mask_points[0]
        mask_points_all.append(np.stack((xs,ys,np.ones_like(xs))).astype(np.float))
        mask_ims.append(cur_mask)
    return mask_points_all, mask_ims

def refine_visual_hull(masks, Ps, scale, center, MINIMAL_VIEWS=45):
    # num_cam=masks.shape[0]
    num_cam = len(masks)
    GRID_SIZE=100
    # MINIMAL_VIEWS=45 # Fitted for DTU, might need to change for different data.
    xx, yy, zz = np.meshgrid(np.linspace(-scale, scale, GRID_SIZE), np.linspace(-scale, scale, GRID_SIZE),
                             np.linspace(-scale, scale, GRID_SIZE))
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()))
    points = points + center[:, np.newaxis]
    appears = np.zeros((GRID_SIZE*GRID_SIZE*GRID_SIZE, 1))
    for i in range(num_cam):
        im_height = masks[i].shape[0]
        im_width = masks[i].shape[1]
        proji = Ps[i] @ np.concatenate((points, np.ones((1, GRID_SIZE*GRID_SIZE*GRID_SIZE))), axis=0)
        depths = proji[2]
        proj_pixels = np.round(proji[:2] / depths).astype(np.long)
        # Find pixels inside the image range (0, width) and (0,height)
        relevant_inds = np.logical_and(proj_pixels[0] >= 0, proj_pixels[1] < im_height)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[0] < im_width)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[1] >= 0)
        # Check if the pixel lies in front of camera (<0 if it is behind camera)
        relevant_inds = np.logical_and(relevant_inds, depths > 0)
        relevant_inds = np.where(relevant_inds)[0]

        # Intersection btw projected cube and mask
        cur_mask = masks[i] > 0.5
        relmask = cur_mask[proj_pixels[1, relevant_inds], proj_pixels[0, relevant_inds]]
        relevant_inds = relevant_inds[relmask]

        appears[relevant_inds] = appears[relevant_inds] + 1

    final_points = points[:, (appears >= MINIMAL_VIEWS).flatten()]
    centroid=final_points.mean(axis=1)
    normalize = final_points - centroid[:, np.newaxis]

    return centroid,np.sqrt((normalize ** 2).sum(axis=0)).mean() * 3,final_points.T

# the normaliztion script needs a set of 2D object masks and camera projection matrices
# (P_i=K_i[R_i |t_i] where [R_i |t_i] is world to camera transformation)
def get_normalization_function(Ps,mask_points_all,number_of_normalization_points,number_of_cameras,masks_all):
    P_0 = Ps[0]
    Fs = get_fundamental_matrices(P_0, Ps)
    P_0_center = np.linalg.svd(P_0)[-1][-1, :]
    P_0_center = P_0_center / P_0_center[3]

    # Use image 0 as a references
    xs = mask_points_all[0][0, :]
    ys = mask_points_all[0][1, :]

    counter = 0
    all_Xs = []

    # sample a subset of 2D points from camera 0
    indss = np.random.permutation(xs.shape[0])[:number_of_normalization_points]
    # Todo, get contour and inner pts

    for i in indss:
        curx = xs[i]
        cury = ys[i]
        # for each point, check its min/max depth in all other cameras.
        # If there is an intersection of relevant depth keep the point
        observerved_in_all = True
        max_d_all = 1e10
        min_d_all = 1e-10
        for j in range(1, number_of_cameras, 5):
            min_d, max_d = get_min_max_d(curx, cury, Ps[j], mask_points_all[j], P_0, Fs[j], j)

            if abs(min_d) < 0.00001:
                observerved_in_all = False
                break
            max_d_all = np.min(np.array([max_d_all, max_d]))
            min_d_all = np.max(np.array([min_d_all, min_d]))
            if max_d_all < min_d_all + 1e-2:
                observerved_in_all = False
                break
        if observerved_in_all:
            direction = np.linalg.inv(P_0[:3, :3]) @ np.array([curx, cury, 1.0])
            all_Xs.append(P_0_center[:3] + direction * min_d_all)
            all_Xs.append(P_0_center[:3] + direction * max_d_all)
            counter = counter + 1

    print("Number of points:%d" % counter)
    centroid = np.array(all_Xs).mean(axis=0)
    # mean_norm=np.linalg.norm(np.array(allXs)-centroid,axis=1).mean()
    scale = np.array(all_Xs).std()

    # OPTIONAL: refine the visual hull
    # centroid,scale,all_Xs = refine_visual_hull(masks_all, Ps, scale, centroid, MINIMAL_VIEWS=45)

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = centroid[0]
    normalization[1, 3] = centroid[1]
    normalization[2, 3] = centroid[2]

    normalization[0, 0] = scale
    normalization[1, 1] = scale
    normalization[2, 2] = scale
    return normalization,all_Xs

def map_intrinsic_models(model_name, params):
    K = np.zeros((3,3))
    K[2,2] = 1.
    if 'SIMPLE' in model_name:
        K[0, 0] = params[0]  # f
        K[1, 1] = params[0]  # f
        K[0, 2] = params[1]  # cx
        K[1, 2] = params[2]  # cy
        dist_coefs= params[3:]
    else:
        K[0, 0] = params[0]  # fx
        K[1, 1] = params[1]  # fy
        K[0, 2] = params[2]  # cx
        K[1, 2] = params[3]  # cy
        dist_coefs = params[4:] if 'PINHOLE' not in model_name else []

    return K, dist_coefs

def get_normalization(source_dir, use_linear_init=False):
    print('Preprocessing', source_dir)

    if use_linear_init:
        #Since there is noise in the cameras, some of them will not apear in all the cameras, so we need more points
        number_of_normalization_points=1000
        cameras_filename = "cameras_linear_init_colmap"
    else:
        number_of_normalization_points = 1000
        cameras_filename = "cameras_blender"

    masks_dir=os.path.join(source_dir, 'mask')
    # make_mask = False
    # if os.path.isdir(masks_dir):
    #     if len(os.listdir(masks_dir))==0:
    #         make_mask = True
    # else:
    #     make_mask = True
    # if make_mask:
    #     images_dir='{0}/image'.format(source_dir)
    #     create_pseudo_mask(images_dir, masks_dir)

    cameras=np.load('{0}/{1}.npz'.format(source_dir, cameras_filename))

    mask_points_all,masks_all=get_all_mask_points(masks_dir)
    number_of_cameras = len(masks_all)
    Ps = get_Ps(cameras, number_of_cameras)

    normalization,all_Xs=get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras,masks_all)

    cameras_new={}
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d'%i]=normalization
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i],np.array([[0,0,0,1.0]])),axis=0).astype(np.float32)

    np.savez('{0}/{1}.npz'.format(source_dir, '_'.join(cameras_filename.split('_')[:-1])), **cameras_new)

    print(normalization)
    print('--------------------------------------------------------')

    if True: #for debugging
        debug_dir = os.path.join(source_dir, 'debug')
        os.makedirs(debug_dir,exist_ok=True)
        masks = sorted(os.listdir(masks_dir))
        for i in range(number_of_cameras):
            plt.figure()

            plt.imshow(mpimg.imread(os.path.join(masks_dir, masks[i])))
            xy = (Ps[i,:2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T)) / (
                        Ps[i,2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T))

            plt.plot(xy[0, :], xy[1, :], '*')
            plt.savefig(os.path.join(debug_dir, f'debug_{i:03d}'))
            # plt.show()

def get_rpdata(rp_root, rp_name, rp_name_, source_dir):
    cam_dir = os.path.join(rp_root, "Camera_Parameter")
    image_dir = os.path.join(rp_root, "Image_GT", rp_name, rp_name_, "image")
    mask_dir = os.path.join(rp_root, "Image_GT", rp_name, rp_name_, "mask")


    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'reproj'), exist_ok=True)

    # Get view lists
    cam_names = [name for name in os.listdir(cam_dir) if ".pkl" in name]
    names = [name[:3] for name in cam_names]

    # For debug purpose
    mesh = o3d.io.read_triangle_mesh(os.path.join(rp_root, "Image_GT", rp_name,
                                                  rp_name_.split('/')[0], rp_name+f"_{rp_name_.split('/')[0]}.obj"))
    pts_ = -np.asarray(mesh.vertices) #To match opencv(colmap) coordinate system
    # pts_ = pts_ @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64).T
    # Load cam parameters
    cam_dict = dict()
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    for name in names:
        cam_path = os.path.join(cam_dir, name+"_cam.pkl")
        img_path = os.path.join(image_dir, name+".png")
        mask_path = os.path.join(mask_dir, name+"0001.png")

        with open(cam_path, 'rb') as f:
            cam_data = pickle.load(f)

        K = cam_data['K'].copy()
        RT = cam_data['RT'].copy()
        M = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=RT.dtype)
        RT[:3, :3] = -RT[:3, :3] @ M
        m = K@RT
        cam_dict['camera_mat_{}'.format(int(name))] = K
        cam_dict['camera_mat_inv_{}'.format(int(name))] = np.linalg.inv(K)
        cam_dict['world_mat_{}'.format(int(name))] = m
        cam_dict['world_mat_inv_{}'.format(int(name))] = np.linalg.inv(np.concatenate([m, bottom], 0))
        cam_dict['scale_mat_{}'.format(int(name))] = np.eye(4)
        cam_dict['scale_mat_inv_{}'.format(int(name))] = np.eye(4)

        with Image.open(img_path) as image:
            img_ = np.array(image)
            cv2.imwrite(os.path.join(source_dir, 'image', 'img_{0:03}{1}'.format(int(name), ".png")),
                        cv2.cvtColor(img_, cv2.COLOR_RGB2BGR))
        with Image.open(mask_path) as image:
            mask_ = np.array(image)
            cv2.imwrite(os.path.join(source_dir, 'mask', 'img_{0:03}{1}.png'.format(int(name), ".png")),
                        cv2.cvtColor(mask_, cv2.COLOR_RGB2BGR))

        # Debug
        if True:
            xy = (m[:2, :] @ np.concatenate((pts_, np.ones((pts_.shape[0], 1))), axis=1).T) / (
                    m[2, :] @ np.concatenate((pts_, np.ones((pts_.shape[0], 1))), axis=1).T)
            plt.figure()
            plt.imshow(img_)
            plt.plot(xy[0, :], xy[1, :], '*')
            # plt.show(block=False)
            plt.savefig(os.path.join(source_dir, 'reproj', f'img_{int(name):03}.png'))


    np.savez('{0}/{1}.npz'.format(source_dir, "cameras_blender"), **cam_dict)

if __name__ == "__main__":
    # In the paper (L. Yariv, 2021, Volume Rendering of Neural Implicit Surfaces)
    # Sec. B.1 In supplement
    # "We used the known camera poses to shift the coordinate system, locating the object at the origin." ...
    # "We further apply a global scale of 3/R_max*1.1 to place all camera centers inside a sphere of radius 3 ... (DTU.py)
    # source_dir = "/ssd2/swheo/db/DTU/scan65"
    rp_root = "/ssd2/swheo/db/RPDataset"
    cam_dir = os.path.join(rp_root, "Camera_Parameter")
    image_dir = os.path.join(rp_root, "Image_GT")
    rp_name = "caren001"
    rp_name_ = "0A/goegap_2k"
    source_dir = f"../data/{rp_name}"
    get_rpdata(rp_root, rp_name, rp_name_, source_dir)
    #
    use_linear_init = False
    get_normalization(source_dir, use_linear_init)

    print('Done!')