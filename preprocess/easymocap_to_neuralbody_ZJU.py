import os
import sys
import json
import cv2
import glob

import numpy as np
import torch
# sys.path.append("../")
from zju_smpl.body_model import SMPLlayer
import tqdm
def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def get_cams(root_dir, exclude=[]):
    intri = cv2.FileStorage(os.path.join(root_dir, 'intri.yml'), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(os.path.join(root_dir, 'extri.yml'), cv2.FILE_STORAGE_READ)
    namenode = extri.getNode('names')
    names = [namenode.at(i).string() for i in range(namenode.size())]
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for c_idx, name in enumerate(names):
        if c_idx + 1 in exclude:
            continue
        cams['K'].append(intri.getNode(f'K_{name}').mat())
        cams['D'].append(intri.getNode(f'dist_{name}').mat().T)
        cams['R'].append(extri.getNode(f'Rot_{name}').mat())
        cams['T'].append(extri.getNode(f'T_{name}').mat())
    return cams


def get_img_paths(root_dir, exclude=[]):
    all_ims = []
    names = sorted(os.listdir(os.path.join(root_dir, 'images')))
    for c_idx, name in enumerate(names):
        if c_idx + 1 in exclude:
            continue
        image_root = os.path.join(root_dir, 'images', name)
        ims = glob.glob(os.path.join(image_root, '*.jpg'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims

def preprocess_easymocap(path, type, save_param=True, save_vertice=True):
    assert type=='smpl', 'currently, models other than smpl is not supported :p'
    smplparam_readdir = os.path.join(path, 'output', type, 'smpl')
    params_outdir, vertices_outdir = None, None
    if save_param:
        params_outdir = os.path.join(path, 'params')
        os.makedirs(params_outdir, exist_ok=True)
    if save_vertice:
        vertices_outdir = os.path.join(path, 'vertices') if save_vertice else None
        os.makedirs(vertices_outdir, exist_ok=True)
        # create smpl model
        model_folder = '/ssd2/swheo/dev/code/neurecon/assets/smpl/'
        device = torch.device('cpu')
        body_model = SMPLlayer(os.path.join(model_folder, 'smpl'),
                               gender='neutral',
                               device=device,
                               regressor_path=os.path.join(model_folder,
                                                           'J_regressor_body25.npy'))
        body_model.to(device)
        bbox = np.zeros((2,3), dtype=np.float32)
        bbox[0,:] = np.infty    # minimum
        bbox[1,:] = -np.infty   # maximum

    smplparam_names = [name for name in os.listdir(smplparam_readdir) if name.endswith('.json')]
    pbar = tqdm.tqdm(smplparam_names, desc='extracting easymocap output...')
    for param_name in pbar:
        file_num = int(param_name.split('.')[0])
        file_name = '%06d' % (file_num+1)
        with open(os.path.join(smplparam_readdir, param_name), 'r') as f:
            ezmocap_params = json.load(f)[0]
        params = {'poses': np.array(ezmocap_params['poses']),
                  'Rh': np.array(ezmocap_params['Rh']),
                  'Th': np.array(ezmocap_params['Th']),
                  'shapes': np.array(ezmocap_params['shapes'])}

        # save parameters
        if save_param:
            np.save(os.path.join(params_outdir,f'{int(file_name)-1}.npy'), params)

        if save_vertice:
            # pose model and save vertices
            vertices = body_model(return_verts=True,
                                  return_tensor=False,
                                  new_params=True,
                                  **params)
            bbox[0, :] = np.min(np.vstack([vertices[0], bbox[0, :]]), axis=0)
            bbox[1, :] = np.max(np.vstack([vertices[0], bbox[1, :]]), axis=0)
            vertices_path = os.path.join(vertices_outdir, f'{int(file_name)-1}.npy')
            np.save(vertices_path, vertices[0])

    cams = get_cams(path, [])
    # # Normalize camera matrics and compute c2w
    centroid = np.array(bbox).mean(axis=0)
    scale = np.array(bbox).std()
    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = centroid[0]
    normalization[1, 3] = centroid[1]
    normalization[2, 3] = centroid[2]

    # Larger than SMPL
    scale_factor = 1.1 # -1/+1
    normalization[0, 0] = scale / scale_factor
    normalization[1, 1] = scale / scale_factor
    normalization[2, 2] = scale / scale_factor

    cam_dict = {}
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    for idx in range(len(cams['K'])):
        K = cams['K'][idx]
        R = cams['R'][idx]
        T = cams['T'][idx]

        m = K@np.concatenate([R, T], 1)
        cam_dict['camera_mat_{}'.format(idx)] = K
        cam_dict['camera_mat_inv_{}'.format(idx)] = np.linalg.inv(K)
        cam_dict['world_mat_{}'.format(idx)] = m
        cam_dict['world_mat_inv_{}'.format(idx)] = np.linalg.inv(np.concatenate([m, bottom], 0))
        cam_dict['dist_mat_{}'.format(idx)] = cams['D'][idx]
        cam_dict['scale_mat_{}'.format(idx)] = normalization

    annot = {}
    annot['cams'] = cams

    img_paths = get_img_paths(path, [])
    ims = []
    for img_path in img_paths:
        data = {}
        data['ims'] = img_path.tolist()
        ims.append(data)
    annot['ims'] = ims
    # cam_dict['img_patt'] = annot['ims']
    np.savez(os.path.join(path, 'cameras.npz'), **cam_dict)
    np.save(os.path.join(path, 'annots.npy'), annot)


if __name__=="__main__":
    dataroot = "/ssd2/swheo/db/ZJU_MOCAP/LightStage"
    subject = "377"
    type = "smpl"

    preprocess_easymocap(os.path.join(dataroot, subject), type, save_param=True, save_vertice=True)