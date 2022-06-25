import cv2
import numpy as np
import torch as T

import scipy.sparse
from scipy.sparse.linalg import spsolve
import imageio
import pickle

from utils.geometry import texcoord2imcoord

class TensorProperties:
    def to(self, device):
        raise NotImplementedError

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

def get_boundary_mask(mask, type=0):
    assert type in [0,1], "get_boundary_mask: support for 4-connect (type=0) or 8-connect (type=1) neighbors"
    k = np.zeros((3,3), dtype=np.uint8)
    if type:
        k[:] = 1
    else:
        k[1] = 1
        k[:,1] = 1

    boundary = mask - cv2.erode(mask.astype(np.uint8) * 255, k) / 255.

    return boundary

def compute_edge_from_mesh(f):
    edge_ , _ = sorted_edge_and_dir(f)
    f_edges_, f_tmp = np.unique(
        edge_,
        axis=0, return_inverse=True)
    f_edges = np.zeros((f_edges_.shape[0], 5), dtype=np.int32)
    f_edges[:, :2] = f_edges_
    f_edges[:, 2:4] = -1
    f_tmp = f_tmp.reshape((3, -1)).transpose()
    for idx, inv_tri in enumerate(f_tmp):
        f_edges[inv_tri, f_edges[inv_tri, 4] + 2] = idx
        f_edges[inv_tri, 4] += 1

    # f_edges[i, :2] : unique combination of edge defined by vertex indices
    # f_edges[i, 2:4] : Face (Triangle) indices which contain f_edges[i, :2] (at Most 2)
    # f_edges[i, 4] How many face contain the i'th edge (if 1, it is on boundary)
    return f_edges[:,:2], f_edges[:, 2:4]

def sorted_edge_and_dir(f):
    # Return either 1 or 0 for dir
    edge = np.vstack([f[:, :2], f[:, 1:], np.hstack([f[:, 2][:, None], f[:, 0][:, None]])])
    sorted_idx = np.argsort(edge, axis=1)
    edge_dir = (sorted_idx == [0, 1])[:, 0]
    edge_sorted = np.vstack([edge[np.arange(0, edge.shape[0]), sorted_idx[:, 0]],
                   edge[np.arange(0, edge.shape[0]), sorted_idx[:, 1]]]).transpose()

    return edge_sorted, edge_dir

def make_seg_gcgraph(iso_mask, s_edges_from, s_edges_to):
    height, width = iso_mask.shape
    assert height == width, "Currently supprot for square texture map, check input or code"
    dr_v = cv2.filter2D(
        cv2.copyMakeBorder(iso_mask, 0, 0, 0, 1, cv2.BORDER_CONSTANT),
        -1, cv2.flip(np.asarray([[-1, 1]]), -1), borderType=cv2.BORDER_CONSTANT)[:, 1:]
    dr_h = cv2.filter2D(
        cv2.copyMakeBorder(iso_mask, 0, 1, 0, 0, cv2.BORDER_CONSTANT),
        -1, cv2.flip(np.asarray([[-1], [1]]), -1), borderType=cv2.BORDER_CONSTANT)[1:,:]

    where_v = iso_mask - dr_v
    where_h = iso_mask - dr_h

    idxs = np.arange(height*width).reshape(height, width)
    v_edges_from = idxs[:-1, :][where_v[:-1, :] == 1].flatten()
    v_edges_to = idxs[1:, :][where_v[:-1, :] == 1].flatten()
    h_edges_from = idxs[:, :-1][where_h[:, :-1] == 1].flatten()
    h_edges_to = idxs[:, 1:][where_h[:, :-1] == 1].flatten()

    edges_from = np.r_[v_edges_from, h_edges_from, s_edges_from]
    edges_to = np.r_[v_edges_to, h_edges_to, s_edges_to]
    edges_w = np.r_[np.ones_like(v_edges_from), np.ones_like(h_edges_from), np.ones_like(s_edges_from)]

    edges = {
        'from': edges_from,
        'to': edges_to,
        'w' : edges_w
    }
    return edges

def compute_seam_edge(iso_mask, vt, ft, f, debug=None):
    # Find duplicates
    v_edge, v_edge_f = compute_edge_from_mesh(f) # returns sorted edges and its corresponding face idx (_f)
    vt_edge, vt_edge_f = compute_edge_from_mesh(ft)
    bdedges = np.where(vt_edge_f[:, 1] == -1)[0] # Edges where is contained only in a single face
    # Find coressponding edge from the boundery edges (bdedge)
    bdedgeto = []
    for bdedge in bdedges:
        # 1. For bd edge bi, find index of bi from triangle f_fidx that contains bi
        fidx = vt_edge_f[bdedge, 0]
        fedges, fdirs = sorted_edge_and_dir(ft[fidx][None, :])
        triidx = np.where(np.all(fedges == vt_edge[bdedge, :], axis=1))[0]
        # 2. For the same triangle in mesh f_fidx, find the adjacent triangle that contain bi's corresponding edge ei on the mesh.
        # There are two triangles containing ei, adjacent triangle would be not the triangle indexed by fidx
        ff, _ = sorted_edge_and_dir(f[fidx][None, :])
        adj = v_edge_f[np.all(v_edge == ff[triidx], axis=1)]
        adj = adj[adj != fidx]
        # 3. For the identified triangle containing ei (which is not f_fidx), find the edge in uv domain which correspond to ei
        # But it is should be not bi
        ff_target, _ = sorted_edge_and_dir(f[adj])
        triidx_target = np.where(np.all(ff_target == ff[triidx], axis=1))
        ft_target, ft_target_dirs = sorted_edge_and_dir(ft[adj])
        bdedgeto.append(np.where(np.all(vt_edge == ft_target[triidx_target], axis=1))[0])
    bdedgeto = np.array(bdedgeto).squeeze()
    seams = np.vstack([bdedges, bdedgeto]).transpose()

    vt_ = texcoord2imcoord(T.from_numpy(vt.astype(np.float32)), iso_mask.shape[0], iso_mask.shape[1]).cpu().numpy()
    iso_boundary = get_boundary_mask(iso_mask, 1)

    # Map edge vt to closest pont on boundary
    iso_bdmask = iso_boundary == 1.
    iso_idx = np.asarray(np.where(iso_bdmask)).transpose() # Note, (y,x) ordering
    vt_bdedge = vt_[vt_edge[bdedges].reshape(-1)].squeeze()
    dists = np.linalg.norm(vt_bdedge[:, None, :] - iso_idx[None, :, ::-1], axis=-1)
    iso_bd = iso_idx[np.argmin(dists, axis=1), :]#.reshape(bdedges.shape[0], -1)
    # vt_bdiso = iso_idx[]
    iso_bdmask[iso_bd[:,0], iso_bd[:,1]] = False
    numbd, bdmap, bdstats, _ = cv2.connectedComponentsWithStats(iso_bdmask.astype(np.uint8), connectivity=4)
    bdstats[:, 2:4] = bdstats[:, 0:2] + bdstats[:, 2:4] - 1
    iso_bd = iso_bd.reshape(-1, 4)
    edge_idx = []
    seams_ = seams.copy()
    test_ = np.zeros_like(bdmap)
    for idx, edges in enumerate(iso_bd):
        edge = edges.reshape(-1, 2)
        x0, y0 = np.min(edge, axis=0)
        x1, y1 = np.max(edge, axis=0)
        tmpl = bdmap[x0:(x1+1), y0:(y1+1)]
        labels, cnts = np.unique(tmpl, return_counts=True)
        cnts[labels==0] = 0
        label = labels[np.argmax(cnts)]
        if cnts.shape[0] < 2:
            tmpmap = np.zeros(bdmap.shape, dtype=np.bool)
        else:
            tmpmap = bdmap == label
        tmpmap[edge[:, 0], edge[:, 1]] = True
        test_[tmpmap == True] = idx+1
        pxs = np.asarray(np.where(tmpmap))
        ### Sorting clock-wise
        order = []
        pxs_ = pxs.copy()
        start = np.argmin(pxs_[1, :])  # start with left-most
        chk_dup = np.where(pxs_[1,:] == pxs_[1, start])
        if len(chk_dup)>1:
            cnt_ = 0
            for start_ in chk_dup:
                if np.sum(np.linalg.norm(pxs_ - pxs_[:, start_][:, None], axis=0) < (1.+1e-6)) < 3:
                    cnt_ += 1
                    start = start_
            if cnt_ == 2:
                start = np.argmax(pxs_[0,:]) # Bottom-most
            # if cnt_ > 2:
            #     foo = 1
        order.append(start)
        while ~np.all(pxs_ == -1):
            pts = pxs_[:, start].copy()
            pxs_[:, start] = -1
            adj = np.argsort(np.linalg.norm(pxs_ - pts[:, None], axis=0))
            start = adj[0]
            order.append(start)

        edge_idx.append((pxs[0, order[:-1]], pxs[1, order[:-1]]))
        seams_[seams == bdedges[idx]] = idx

    # Sanity check
    if debug != None:
        import os
        os.makedirs(debug, exist_ok=True)
        for idx in range(seams_.shape[0]):
            i1, i2 = seams_[idx, :]

            edge_line1, edge_line2 = np.zeros(iso_mask.shape), np.zeros(iso_mask.shape)
            # edge_line1[edge_idx[i1][0], edge_idx[i1][1]] = True
            edge_line1[edge_idx[i1][0], edge_idx[i1][1]] = np.arange(1, edge_idx[i1][0].shape[0] + 1)
            # edge_line2[edge_idx[i2][0], edge_idx[i2][1]] = True
            edge_line2[edge_idx[i2][0], edge_idx[i2][1]] = np.arange(1, edge_idx[i2][0].shape[0] + 1)
            tmp = np.stack([edge_line1, edge_line2, iso_mask * 0.2], axis=2)

            cv2.imwrite(os.path.join(debug, 'pair_{idx}.png'), np.uint8(255 * tmp))


    # Note, the direction of seams_ is opposite each other (i.e., direction(edge[seam[0,0]]) = -direction(edge[seam[0,1]]))
    # reduced pixel indices on iso map / corresponding seam connectivity / total edges / original seam idx matches / vt_
    return edge_idx, seams_, vt_edge, seams, vt_

def edges_seams(seams, tex_res, edge_idx):
    edges = np.zeros((0, 2), dtype=np.int32)

    for _, e0, _, e1 in seams:
        idx0 = np.array(edge_idx[e0][0]) * tex_res + np.array(edge_idx[e0][1])
        idx1 = np.array(edge_idx[e1][0]) * tex_res + np.array(edge_idx[e1][1])
        idx1 = idx1[::-1]
        if len(idx0) and len(idx1):
            if idx0.shape[0] < idx1.shape[0]:
                idx0 = cv2.resize(idx0.reshape(-1, 1), (1, idx1.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif idx0.shape[0] > idx1.shape[0]:
                idx1 = cv2.resize(idx1.reshape(-1, 1), (1, idx0.shape[0]), interpolation=cv2.INTER_NEAREST)

            edges_new = np.hstack((idx0.reshape(-1, 1), idx1.reshape(-1, 1)))
            edges = np.vstack((edges, edges_new))

    edges = np.sort(edges, axis=1)
    edges = edges[~(edges[:, 0] == edges[:, 1]), :]
    return edges[:, 0], edges[:, 1]

def laplacian_matrix(n, m):
    """Generate the Poisson matrix.
    Refer to:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    mat_C = mat_A.tocoo()
    return mat_A, np.vstack((mat_C.data, mat_C.row, mat_C.col)).transpose()

def pre_compute_poisson_vars(im_size, offset = (0, 0), edges=None):
    # im_size = (height, width)
    y_max, x_max = im_size
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min

    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])

    mat_A, ijv = laplacian_matrix(y_range, x_range)
    # separate ijv into its corresponding diagonal components
    # [0, 1, -1, xrange, -xrange]

    d_offsets = [0, 1, -1, x_range, -x_range]

    values = [0, -1, -1, -1, -1, -1, -1] # Last one for edge connection
    # values = [0, -1, -1, -1, -1]  # Last one for edge connection
    # values = [4, -1, -1, -1, -1]  # Sanity check
    idxes = np.zeros((len(values), x_range * y_range), dtype=np.int64) - 1
    for i, offset in enumerate(d_offsets):
        d_set = ijv[(ijv[:, 1] + offset) == ijv[:, 2], :]
        idxes[i, d_set[:, 1].astype(np.int64)] = d_set[:, 2].astype(np.int64)
    for i in range(edges[0].shape[0]):
        if edges[1][i] not in idxes[:, edges[0][i]]:
            idxes[-1, edges[0][i]] = edges[1][i]
        if edges[0][i] not in idxes[:, edges[1][i]]:
            idxes[-2, edges[1][i]] = edges[0][i]

    laplacian = idxes_to_csc(values, idxes)

    p = {'M':M,
        'mat_A':mat_A,
        'ijv': ijv,
        'values': values,
        'idxes': idxes,
        'laplacian': laplacian
    }

    return p

def idxes_to_csc(values, idxes, ijv_in=None):
    ijv_ = np.zeros((0, 3), dtype=np.float32)
    rows = np.arange(0, idxes.shape[1])
    msks = idxes != -1
    for value, cols, msk in zip(values, idxes, msks):
        msk = cols != -1
        row = rows[msk]
        col = cols[msk]
        if value == 0:
            # Main diagonal
            val = np.int64(np.sum(msks[:, msk], axis=0) - 1)
        else:
            val = np.ones_like(row)*value
        vrc = np.vstack([val, row, col]).transpose()
        ijv_ = np.vstack([ijv_, vrc])
    if np.all(ijv_in != None):
        ijv_ = np.vstack([ijv_, ijv_in])

    mat = scipy.sparse.csc_matrix((ijv_[:, 0], (ijv_[:, 1].astype(np.int64), ijv_[:, 2].astype(np.int64))))

    return mat


import time as t
def poisson_edit(source, target, mask, poisson_p):
    """The poisson blending function.
    Refer to:
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume:
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min

    if poisson_p == None:
        offset = (0,0)
        M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
        mat_A = laplacian_matrix(y_range, x_range)
        laplacian = mat_A.tocsc()
    else:
        # Use precomputed one
        M = poisson_p['M']
        # mat_A = poisson_p['mat_A'].copy()
        # ijv = poisson_p['ijv'].copy()
        laplacian = poisson_p['laplacian']
        values = poisson_p['values'][:]
        idxes = poisson_p['idxes'].copy()

    source = cv2.warpAffine(source, M, (x_range, y_range))

    mask = mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1
    # mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # for \Delta g
    tmp = mask.copy()
    tmp[:, [0, -1]] = 1
    tmp[[0, -1], :] = 1
    mask_k = (1 - tmp.flatten()).astype(np.bool)

    values.append(1)
    tmp_ = np.arange(0, x_range*y_range)[None, :]
    tmp_[:, ~mask_k] = -1
    idxes[:, mask_k] = -1
    idxes = np.concatenate((idxes, tmp_), axis=0)
    mat_A = idxes_to_csc(values, idxes)

    # start = t.time()
    # mask_flat = mask.flatten()
    mask_flat = tmp.flatten()
    if len(source.shape) == 2:
        source = source[..., None]
    mat_bs = []
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        # concat = source_flat*mask_flat + target_flat*(1-mask_flat)

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat) * alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

        mat_bs.append(mat_b)

    # print(f'step 5 {t.time() - start}, mat_a {mat_A.nnz}, mat_b {np.sum(mat_b!=0)}')

    # start = t.time()
    mat_b_ = np.vstack(mat_bs).transpose()
    x_ = spsolve(mat_A, mat_b_, use_umfpack=True)
    for channel in range(source.shape[2]):
        x = x_[:,channel].reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        target[y_min:y_max, x_min:x_max, channel] = x
    # print(f'step 6 {t.time() - start}, mat_a {mat_A.nnz}, mat_b {np.sum(mat_b!=0)}')
    # print(f'validity check {np.sum(target.astype(np.float64) - target_.astype(np.float64))}')
    return target


def read_obj(file_name):
    vertices = []
    faces = []
    face_textures = []
    textures = []
    try:
        f = open(file_name)
        for line in f:
            if line[:2] == "v ":
                v = line.split()[1:]
                vertices.append(tuple([float(vert) for vert in v]))

            elif line[0] == "f":
                string = line.replace("//", "/")
                ##
                each_vertex = string.split()[1:]
                face, tface, ff, ft = [], [], [], []
                for item in each_vertex:
                    if '/' in item:
                        items = item.split('/')
                        face.append(items[0])
                        tface.append(items[1])
                        ##
                        for v, vt in zip(face, tface):
                            v, vt = int(v), int(vt)
                            ff.append(v)
                            ft.append(vt)
                    else:
                        face.append(item)
                ##
                faces.append(tuple([int(v) for v in face]))
                if len(tface) > 1:
                    face_textures.append(tuple([int(v) for v in tface]))
            elif line[:2] == 'vt':
                vt = line.split()[1:]
                textures.append(tuple([float(t) for t in vt]))
        f.close()
    except IOError:
        print(".obj file not found.")

    return {'v' : np.array(vertices, dtype=np.float32),
            'f' : np.array(faces, dtype=np.int32) - 1,
            'vt' : np.array(textures, dtype=np.float32),
            'ft' : np.array(face_textures, dtype=np.int32) - 1}

if __name__=="__main__":
    uv_map = imageio.imread('../assets/smpl/smpl/smpl_uv.png')
    mesh = read_obj('../assets/smpl/smpl/smpl_uv.obj')
    vt = mesh['vt']
    ft = mesh['ft']
    f = mesh['f']

    iso_mask = np.array(uv_map[...,3]).astype(np.float32)/255.
    edge_idx_, seams_, _, _, _ = compute_seam_edge(iso_mask, vt, ft, f, debug='../assets/smpl/sanity_check')
    seams_ = np.vstack([np.zeros(seams_.shape[0]), seams_[:, 0], np.zeros(seams_.shape[0]), seams_[:, 1]]).astype(
        np.int32).transpose()
    s_edges_from, s_edges_to = edges_seams(seams_, iso_mask.shape[0], edge_idx_)
    edges = make_seg_gcgraph(iso_mask, s_edges_from, s_edges_to)
    foo =1