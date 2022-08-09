import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
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

    for i in indss:
        curx = xs[i]
        cury = ys[i]
        # for each point, check its min/max depth in all other cameras.
        # If there is an intersection of relevant depth keep the point
        observerved_in_all = True
        max_d_all = 1e10
        min_d_all = 1e-10
        for j in range(1, number_of_cameras, 1):
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


def create_pseudo_mask(images_dir, write_to):
    print("No mask available creating it to {0}".format(write_to))
    os.makedirs(write_to, exist_ok=True)
    img_paths = sorted(glob_imgs(images_dir))
    for path in img_paths:
        img = mpimg.imread(path)
        name = path.split('/')[-1]
        mask = np.sum(img**2, axis=2)>0.0
        out = np.zeros_like(img).astype(np.float)
        out[mask,:] = 1
        mpimg.imsave(os.path.join(write_to, name + ".png"), out)

def process_colmap(colmap_dir, write_to, images_dir, mask_dir, write_yml=True):
    camerasfile = os.path.join(colmap_dir, 'sparse/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    imagesfile = os.path.join(colmap_dir, 'sparse/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    pointsfile = os.path.join(colmap_dir, 'sparse/points3D.bin')
    ptdata = read_model.read_points3d_binary(pointsfile)

    procssed_dir = os.path.join(colmap_dir, 'processed')
    os.makedirs(procssed_dir, exist_ok=True)
    os.makedirs(os.path.join(procssed_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(procssed_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(procssed_dir, 'semantic'), exist_ok=True)
    os.makedirs(os.path.join(procssed_dir, 'reproj'), exist_ok=True)

    cam_dict = dict()

    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    names = [imdata[k].name.split('.')[0] for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    permkeys = np.array(list(imdata.keys()))[perm]
    # Set maximum size to make all image have same size
    maxH, maxW = 0, 0
    for c in camdata:
        cam = camdata[c]
        width, height = cam.width, cam.height
        maxW = width if width > maxW else maxW
        maxH = height if height > maxH else maxH
    maskdirs = sorted(os.listdir(mask_dir))
    maskdirs = [maskdir for maskdir in maskdirs if maskdir.split('.')[0] in names]

    ptsall = []
    for pt in ptdata:
        ptsall.append(ptdata[pt].xyz)
    ptsall = np.array(ptsall)

    if write_yml:
        intri = cv2.FileStorage(os.path.join(procssed_dir, 'intri.yml'), cv2.FILE_STORAGE_WRITE)
        extri = cv2.FileStorage(os.path.join(procssed_dir, 'extri.yml'), cv2.FILE_STORAGE_WRITE)
        intri.write('names', sorted(names))
        extri.write('names', sorted(names))

    for i, k in enumerate(permkeys):
        im = imdata[k]
        # Check image size
        cam = camdata[im.camera_id]
        ratioW, ratioH = maxW/cam.width, maxH/cam.height

        # Note. Colmap sometimes remove some iamges
        with Image.open(os.path.join(images_dir, im.name)) as image:
            img_orig = np.array(image)
            if np.any([ratioW>1.0, ratioH>1.0]):
                target_shape = (maxH, maxW, 3)
                scaleW, scaleH, txf, tyf = center_resize_info(img_orig, target_shape)
                img_resize = np.zeros(target_shape, dtype=img_orig.dtype)
                image_ = cv2.resize(img_orig.copy(), dsize=None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_LINEAR)
                A = np.array([[scaleW, 0, txf], [0, scaleH, tyf], [0, 0, 1]])
                img_resize[tyf:(tyf+image_.shape[0]), txf:(txf+image_.shape[1]),...] = image_
            else:
                img_resize = img_orig
                A = np.eye(3)
            cv2.imwrite(os.path.join(procssed_dir, 'image', 'img_{0:03}.{1}'.format(i, im.name.split('.')[-1])), cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR))

        with Image.open(os.path.join(mask_dir, maskdirs[i])) as image:
            mask_orig = np.array(image)
            target_shape = (maxH, maxW, 4)
            if np.any([ratioW > 1.0, ratioH > 1.0]):
                mask_resize = np.zeros(target_shape, dtype=img_orig.dtype)
                mask_ = cv2.resize(mask_orig.copy(), dsize=None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_NEAREST)
                mask_resize[tyf:(tyf + mask_.shape[0]), txf:(txf + mask_.shape[1]), ...] = mask_
            else:
                mask_resize = mask_orig
            if len(mask_resize.shape)==2:
                mask_resize = np.repeat(mask_resize[...,None], 3, -1)
            # mask_resize[np.sum(mask_resize**2, axis=2)>0.0, :] = 255 if mask_resize.dtype==np.uint8 else 1
            # mask_resize[mask_resize> 0.0] = 255 if mask_resize.dtype == np.uint8 else 1
            cv2.imwrite(os.path.join(procssed_dir, 'mask', 'img_{0:03}.{1}.png'.format(i, im.name.split('.')[-1])), cv2.cvtColor(mask_resize, cv2.COLOR_RGB2BGR))

            # Make semantic labels
            labels_color = np.unique(mask_resize.reshape(-1, mask_resize.shape[-1]), axis=0)
            if labels_color.shape[0] > 2:
                labels = np.zeros((labels_color.shape[0], ), dtype=np.int)
                semantics = np.zeros(mask_resize.shape[:-1], dtype=np.uint8)
                from dataio.colormap import CIHP20, NULB_CMAP
                for k, v in NULB_CMAP.items():
                    for id in v:
                        labels[np.all(labels_color==CIHP20[id][::-1], axis=-1)] = k
                for label, label_color in zip(labels, labels_color):
                    semantics[np.all(mask_resize==label_color, axis=-1)] = label
                cv2.imwrite(os.path.join(procssed_dir, 'semantic', 'img_{0:03}.{1}.png'.format(i, im.name.split('.')[-1])),
                            semantics)

        K, _ = map_intrinsic_models(camdata[im.camera_id].model, camdata[im.camera_id].params)
        K = A@K #np.array([[ratioW, 0, 0], [0, ratioH, 0], [0, 0, 1]])@K
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = K@np.concatenate([R, t], 1)
        if write_yml:
            intri.write(f'K_{names[k]}', K)
            intri.write(f'dist_{names[k]}', _ if len(_)>0 else np.zeros((5,1)))
            extri.write(f'Rot_{names[k]}', R)
            extri.write(f'R_{names[k]}', cv2.Rodrigues(R)[0])
            extri.write(f'T_{names[k]}', t)
        # Debug
        if True:
            pts_ = []
            for pt in ptdata:
                if im.id in ptdata[pt].image_ids:
                    pts_.append(ptdata[pt].xyz)
            pts_ = np.array(pts_)
            # K_, _ = map_intrinsic_models(camdata[im.camera_id].model, camdata[im.camera_id].params)
            # m_ = K_ @ np.concatenate([R, t], 1)
            # xy = (m_[:2, :] @ np.concatenate((pts_, np.ones((pts_.shape[0], 1))), axis=1).T) / (
            #             m_[2, :] @ np.concatenate((pts_, np.ones((pts_.shape[0], 1))), axis=1).T)
            # plt.imshow(img_orig)
            # plt.plot(xy[0, :], xy[1, :], '*')
            # plt.show()
            xy = (m[:2, :] @ np.concatenate((pts_, np.ones((pts_.shape[0], 1))), axis=1).T) / (
                    m[2, :] @ np.concatenate((pts_, np.ones((pts_.shape[0], 1))), axis=1).T)
            plt.figure()
            plt.imshow(img_resize*(mask_resize>1))
            plt.plot(xy[0, :], xy[1, :], '*')
            # plt.show(block=False)
            plt.savefig(os.path.join(procssed_dir, 'reproj', f'img_{i:03}.{im.name.split(".")[-1]}.png'))
        # Ps.append(np.concatenate([m, bottom], 0))
        cam_dict['camera_mat_{}'.format(i)] = K
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(K)
        cam_dict['world_mat_{}'.format(i)] = m
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(np.concatenate([m, bottom], 0))
    cam_dict['pts_all'] = ptsall
    # Ps = np.stack(Ps, 0)
    # Psinv = np.linalg.inv(Ps)

    # Note. From github.com/Fyusion/LLFF/llff/poses/pose_utils.py line 50.
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t] (No need to)
    #
    np.savez(write_to, **cam_dict)
    if write_yml:
        intri.release()
        extri.release()

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
        number_of_normalization_points = 2000
        cameras_filename = "cameras_colmap"

    masks_dir=os.path.join(source_dir, 'mask')
    make_mask = False
    if os.path.isdir(masks_dir):
        if len(os.listdir(masks_dir))==0:
            make_mask = True
    else:
        make_mask = True
    if make_mask:
        images_dir='{0}/image'.format(source_dir)
        create_pseudo_mask(images_dir, masks_dir)

    camera_dir = '{0}/{1}.npz'.format(source_dir, cameras_filename)
    need_process_colmap = False
    if not os.path.exists(camera_dir):
        need_process_colmap = True
    if need_process_colmap:
        images_dir = '{0}/image'.format(source_dir)
        colmap_dir = os.path.join(source_dir, 'colmap')
        source_dir = os.path.join(source_dir, 'colmap/processed')
        camera_dir = '{0}/{1}.npz'.format(source_dir, cameras_filename)
        process_colmap(colmap_dir, camera_dir, images_dir, masks_dir)
        masks_dir = os.path.join(source_dir, 'mask')

    cameras=np.load('{0}/{1}.npz'.format(source_dir, cameras_filename))

    mask_points_all,masks_all=get_all_mask_points(masks_dir)
    number_of_cameras = len(masks_all)
    Ps = get_Ps(cameras, number_of_cameras)

    normalization,all_Xs=get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras,masks_all)

    cameras_new={}
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d'%i]=normalization
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i],np.array([[0,0,0,1.0]])),axis=0).astype(np.float32)
    if 'pts_all' in cameras.keys():
        cameras_new['pts_all'] = cameras['pts_all']
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

if __name__ == "__main__":
    # In the paper (L. Yariv, 2021, Volume Rendering of Neural Implicit Surfaces)
    # Sec. B.1 In supplement
    # "We used the known camera poses to shift the coordinate system, locating the object at the origin." ...
    # "We further apply a global scale of 3/R_max*1.1 to place all camera centers inside a sphere of radius 3 ... (DTU.py)
    # source_dir = "/ssd2/swheo/db/DTU/scan65"
    source_dir = "../data/jmg_cellphone"
    use_linear_init = False
    get_normalization(source_dir, use_linear_init)

    print('Done')