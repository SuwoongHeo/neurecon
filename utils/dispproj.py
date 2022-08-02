import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import trimesh
import torch as T
import numpy as np
import torch.nn.functional as F

from utils.io_util import read_obj
from utils.geometry import Mesh, _approx_inout_sign_by_normals, _approx_inout_sign_raytracing, project2closest_face

from tools.vis_util import plotly_viscorres3D

from pytorch3d import _C
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class DispersedProjector(object):
    """
    Surface-Aligned Neural Radiance Fields for Controllable 3D Human Synthesis, Xu et al, 2022 CVPR
    Modified from original github:
    https://github.com/pfnet-research/surface-aligned-nerf/blob/master/lib/networks/projection/map.py#L49
    """

    def __init__(self, cache_path, mesh=None):
        if not os.path.exists(os.path.join(cache_path, 'faces_to_corres_edges.npy')):
            print(f'Pre-computed topology does not exist. Creating it at {cache_path}')
            if mesh is None:
                mesh = Mesh(file_name=os.path.join(cache_path,'smpl_uv.obj'))
            faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces = self._parse_mesh(mesh.vertices, mesh.faces)
            # save cache
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            np.save(os.path.join(cache_path, 'faces_to_corres_edges.npy'), faces_to_corres_edges.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, 'edges_to_corres_faces.npy'), edges_to_corres_faces.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, 'verts_to_corres_faces.npy'), verts_to_corres_faces.to('cpu').detach().numpy().copy())
            print('==> Finished! Cache saved to: ', cache_path)
        else:
            faces_to_corres_edges = T.from_numpy(np.load(os.path.join(cache_path, 'faces_to_corres_edges.npy')))
            edges_to_corres_faces = T.from_numpy(np.load(os.path.join(cache_path, 'edges_to_corres_faces.npy')))
            verts_to_corres_faces = T.from_numpy(np.load(os.path.join(cache_path, 'verts_to_corres_faces.npy')))

        self.faces_to_corres_edges = faces_to_corres_edges.long()  # [13776, 3]
        self.edges_to_corres_faces = edges_to_corres_faces.long()  # [20664, 2]
        self.verts_to_corres_faces = verts_to_corres_faces.long()  # [6890, 9]

    def __call__(self, query, mesh, vnear=None, st=None, idxs=None, points_inside_mesh_approx=True):
        if len(query.shape) == 3:
            N_ray = 1
            B, N_pts, _ = query.shape
        elif len(query.shape) == 4:
            B, N_ray, N_pts, _ = query.shape
        else:
            N_ray, B = 1, 1
            N_pts, _ = query.shape

        stability_scale = mesh.stability_scale

        vquery = query.view(-1, 3) * stability_scale
        if (vnear is None) and (st is None) and (idxs is None):
            # Step 1: Compute the nearest point s on the mesh surface
            vnear, st, idxs = project2closest_face(vquery, mesh, stability_scale=stability_scale)


        vnear = vnear.view(-1, 3) * stability_scale
        st = st.view(-1, 2)
        idxs = idxs.view(-1)
        # Triangles that containing cloeset points
        tri = mesh.triangles.view(-1, 3, 3) * stability_scale  # [querysize, 3, 3]
        # (s,t) = v0 + s(v1-v0) + t(v2-v0) => (b0,b1,b2) = (1-s-t)v0 + (s)v1 + (t)v2
        bc = T.cat([(1 - T.sum(st, -1)).unsqueeze(-1), st], dim=-1)

        with T.no_grad():
            # st combinations (s+t<=1)
            #        v1
            #       /   \
            #     v0-----v2
            # (0,0) - v0, (0,>0) - e1 - v0v2,
            # (1,0) - v1, (>0, 0) - e2 - v0v1
            # (0,1) - v2, 1-s-t=0, (>0, >0) - e0 - v1v2
            is_nonzero = bc > 0.
            is_one = bc == 1.

            is_a = is_one[..., 0] & ~is_nonzero[..., 1] & ~is_nonzero[..., 2]  # v0
            is_b = is_one[..., 1] & ~is_nonzero[..., 0] & ~is_nonzero[..., 2]  # v1
            is_c = is_one[..., 2] & ~is_nonzero[..., 0] & ~is_nonzero[..., 1]  # v2
            is_bc = ~is_nonzero[..., 0] & is_nonzero[..., 1] & is_nonzero[..., 2] # v1v2
            is_ac = ~is_nonzero[..., 1] & is_nonzero[..., 0] & is_nonzero[..., 2] # v0v2
            is_ab = ~is_nonzero[..., 2] & is_nonzero[..., 0] & is_nonzero[..., 1] # v0v1
            # remain = is_nonzero[...,0] & is_nonzero[..., 1] & is_nonzero[..., 2]  # inside triangle

        if points_inside_mesh_approx:
            sign_ = _approx_inout_sign_by_normals(vquery, vnear, bc, mesh.vert_normal[mesh.faces][idxs])
        else:
            sign_ = _approx_inout_sign_raytracing(vquery, mesh)  # Sanity check

        # Step 2-6: Compute the final projection point
        # diff = vquery - vnear
        # with T.no_grad():
        #     diff = T.sign(diff) * T.clamp(T.abs(diff), min=1e-12)
        #
        # dist = T.norm(diff, dim=-1)
        dist = T.sqrt(T.sum((vquery - vnear) ** 2, -1) + 1e-12) ## make it differentiable
        dist_ = dist.clone()
        vnear_ = vnear.clone()
        idxs_ = idxs.clone()
        # _revise_nearest
        def _revise(is_x, x_idx, x_type):
            query_is_x = vquery[is_x]
            inside_is_x = sign_[is_x]
            if x_type == 'verts':
                verts_is_x = mesh.faces[idxs][is_x][:, x_idx]
                corres_faces_is_x = self.verts_to_corres_faces[verts_is_x] # faces touching the vertex (maximum 9, smpl)
                N_repeat = 9  # maximum # of adjacent faces for verts
            elif x_type == 'edges':
                edges_is_x = self.faces_to_corres_edges[idxs][is_x][:, x_idx]
                corres_faces_is_x = self.edges_to_corres_faces[edges_is_x] # faces touching the edge (maximum 2)
                N_repeat = 2  # maximum # of adjacent faces for edges
            else:
                raise ValueError('x_type should be verts or edges')

            # STEP 2: Find a set T of all triangles containing the vnear (closest point)
            triangles_is_x = tri[corres_faces_is_x]
            verts_normals_is_x = mesh.vert_normal[mesh.faces][corres_faces_is_x]
            faces_normals_is_x = mesh.faces_normal[corres_faces_is_x]

            # STEP 3: Vertex normal alignment
            # with T.no_grad():
            verts_normals_is_x_aligned = self._align_verts_normals(verts_normals_is_x, triangles_is_x, inside_is_x) #vertex normal alignment

            # STEP 4: Check if inside the parallel triangle T'
            points_is_x_repeated = query_is_x.unsqueeze(1).repeat(1, N_repeat, 1)
            inside_control_volume, barycentric = \
                self._calculate_points_inside_target_volume(query_is_x.unsqueeze(1).repeat(1, N_repeat, 1), triangles_is_x,
                                                            verts_normals_is_x_aligned, faces_normals_is_x,
                                                            return_barycentric=True)  # (n', N_repeat):bool, (n', N_repeat, 3)

            # swheo: For the case when the point is not in any parallel triangles (to prevent gradient explosion)
            if T.any(T.all(inside_control_volume == 0, dim=-1)):
                inside_control_volume[T.all(inside_control_volume == 0, dim=-1)] = 1
            barycentric = T.clamp(barycentric, min=0.)
            barycentric = barycentric / (T.sum(barycentric, dim=-1, keepdim=True) + 1e-12)

            # STEP 5: compute set of canditate surface points {s}
            surface_points_set = (barycentric[..., None] * triangles_is_x).sum(dim=2)
            # Note. swheo
            # here is the point where arises gradient explosion / 1e10
            # make gradient explosion check inside_control_volume
            # points mapped outside triangle will be discarded by T.min()
            surface_to_points_dist_set = T.sqrt(
                T.sum((points_is_x_repeated - surface_points_set)**2, dim=2) + 1e-12) + 1e10 * (
                            1 - inside_control_volume)  # [n', N_repeat],
            _, idx_is_x = T.min(surface_to_points_dist_set, dim=1)  # [n', ]

            # STEP 6: Choose the nearest point to x from {s} as the final projection point
            surface_points = surface_points_set[T.arange(len(idx_is_x)), idx_is_x]  # [n', 3]
            surface_to_points_dist = surface_to_points_dist_set[T.arange(len(idx_is_x)), idx_is_x]  # [n', ]
            faces_is_x = corres_faces_is_x[T.arange(len(idx_is_x)), idx_is_x]

            # update (overwrite)
            vnear_[is_x] = surface_points
            dist_[is_x] = surface_to_points_dist
            idxs_[is_x] = faces_is_x

        # revise verts
        if T.any(is_a): _revise(is_a, 0, 'verts')
        if T.any(is_b): _revise(is_b, 1, 'verts')
        if T.any(is_c): _revise(is_c, 2, 'verts')

        # revise edges
        if T.any(is_bc): _revise(is_bc, 0, 'edges')
        if T.any(is_ac): _revise(is_ac, 1, 'edges')
        if T.any(is_ab): _revise(is_ab, 2, 'edges')

        h = dist_ * sign_

        tri_ = mesh.triangles[idxs_].view(-1, 3, 3) * stability_scale #updated triangles after dispersed projection
        bc_ = points_to_barycentric(tri_, vnear_)

        # bad case
        bc_ = T.clamp(bc_, min=0.)
        bc_ = bc_ / (T.sum(bc_, dim=1, keepdim=True) + 1e-12)
        st_ = bc_[...,1:].view((B, N_ray, N_pts, -1)) # barycentric to st

        vnear = vnear.view((B, N_ray, N_pts, -1)) / stability_scale
        vnear_ = vnear_.view((B, N_ray, N_pts, -1)) / stability_scale
        idxs_ = idxs_.view((B, N_ray, N_pts))
        h = h.view((B, N_ray, N_pts, 1)) / stability_scale

        return h, st_, vnear, vnear_, idxs_


    def _align_verts_normals(self, verts_normals, triangles, ps_sign):
        batch_dim = verts_normals.shape[:-2]
        # if batch dim is larger than 1:
        if verts_normals.dim() > 3:
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            ps_sign = ps_sign.unsqueeze(1).repeat(1, batch_dim[1]).view(-1)

        # revert the direction if points inside the mesh
        verts_normals_signed = verts_normals*ps_sign.view(-1, 1, 1)

        # This maybe the point where the unstable gradient (casuing inf) occurs
        edge1 = triangles - triangles[:, [1, 2, 0]]
        edge2 = triangles - triangles[:, [2, 0, 1]]

        # norm edge direction
        edge1_dir = F.normalize(edge1, dim=2)
        edge2_dir = F.normalize(edge2, dim=2)

        # project verts normals onto triangle plane
        faces_normals = T.cross(triangles[:, 0]-triangles[:, 2], triangles[:, 1]-triangles[:, 0], dim=1)
        verts_normals_projected = verts_normals_signed - T.sum(verts_normals_signed*faces_normals.unsqueeze(1), dim=2, keepdim=True)*faces_normals.unsqueeze(1)

        # Algorithm 2 of original paper
        p = T.sum(edge1_dir*verts_normals_projected, dim=2, keepdim=True)
        q = T.sum(edge2_dir*verts_normals_projected, dim=2, keepdim=True)
        r = T.sum(edge1_dir*edge2_dir, dim=2, keepdim=True)

        inv_det = 1 / (1 - r**2 + 1e-9)
        c1 = inv_det * (p - r*q)
        c2 = inv_det * (q - r*p)

        # only align inside normals
        c1 = T.clamp(c1, max=0.)
        c2 = T.clamp(c2, max=0.)

        verts_normals_aligned = verts_normals_signed - c1*edge1_dir - c2*edge2_dir
        verts_normals_aligned = F.normalize(verts_normals_aligned, eps=1e-12, dim=2)

        # revert the normals direction
        verts_normals_aligned = verts_normals_aligned*ps_sign.view(-1, 1, 1)

        return verts_normals_aligned.view(*batch_dim, 3, 3)

    def _calculate_points_inside_target_volume(self, points, triangles, verts_normals, faces_normals, return_barycentric=False):
        batch_dim = points.shape[:-1]
        # if batch dim is larger than 1:
        if points.dim() > 2:
            points = points.view(-1, 3)
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            faces_normals = faces_normals.view(-1, 3)
        dist = ((points-triangles[:, 0]) * faces_normals).sum(1)
        verts_normals_cosine = (verts_normals * faces_normals.unsqueeze(1)).sum(2)  # [batch*65536, 3]
        triangles_parallel = triangles + verts_normals * (dist.view(-1, 1) / (verts_normals_cosine + 1e-12)).unsqueeze(2)  # [batch*13776, 3, 3]
        # todo, without vert alignment, why don't we just move triangle directly?
        barycentric = points_to_barycentric(triangles_parallel, points)
        # Note. swheo: If point is on edge or vertex, it will yield large gradient when computing normal (nabla)
        # so using >= is not valid here (stated below by me)
        # # inside = T.prod(barycentric >= 0, dim=1) #For on-surface points, rather than >, use >=
        inside = T.prod(barycentric > 0, dim=1)

        if return_barycentric:
            return inside.view(*batch_dim), barycentric.view(*batch_dim, -1)
        else:
            return inside.view(*batch_dim)
        ## debugging
        # inside_ = inside.view(*batch_dim)
        # T.where(T.all(inside_ == 0, dim=-1))
        # inside_[9816]
        # tris = triangles.view(*batch_dim, 3, 3)[9816]
        # import plotly.graph_objects as go
        # import plotly.io as pio
        # pio.renderers.default = "browser"
        # tris_plemsh = []
        # for tri in tris:
        #     tris_plmesh.append(go.Mesh3d(x=tri[:, 0].detach().cpu().numpy()), y=tri[:, 1].detach().cpu().numpy(),
        #                        z=tri[:, 2].detach().cpu().numpy(), i=np.array([0]), j=np.array([1]), k=np.array([2]),
        #                        color='grey', lighting=dict(ambient=0.2, diffuse=0.8),
        #                        lightposition=dict(x=0, y=0, z=-1))
        # fig = go.Figure(data=tris_plmesh)
        # fig.show()
        # pts_marker = go.Scatter3d(x=(), y=(pts.detach().cpu().numpy()[9816, 0, 1]),
        #                           z=(pts.detach().cpu().numpy()[9816, 0, 2]),
        #                           marker=go.scatter3d.Marker(size=3, color=np.ones((3, 1)) * 0.8),
        #                           mode='markers')
        # tris_plmesh_ = []
        # for tri in tris_:
        #     tris_plmesh_.append(go.Mesh3d(x=tri[:, 0].detach().cpu().numpy(), y=tri[:, 1].detach().cpu().numpy(),
        #                                   z=tri[:, 2].detach().cpu().numpy(), i=np.array([0]), j=np.array([1]),
        #                                   k=np.array([2]), color='grey', lighting=dict(ambient=0.2, diffuse=0.8),
        #                                   lightposition=dict(x=0, y=0, z=-1)))
        # fig = go.Figure(data=[*tris_plmesh, *tris_plmesh_, pts_marker])
        # fig.show()
        # triangles_parallel_f = triangles + (faces_normals * dist.unsqueeze(-1)).unsqueeze(1)
        # tris_plmesh__ = []
        # tris__ = triangles_parallel_f.view(*batch_dim, 3, 3)[9816]
        # for tri in tris__:
        #     tris_plmesh__.append(go.Mesh3d(x=tri[:, 0].detach().cpu().numpy(), y=tri[:, 1].detach().cpu().numpy(),
        #                                    z=tri[:, 2].detach().cpu().numpy(), i=np.array([0]), j=np.array([1]),
        #                                    k=np.array([2]), color='grey', lighting=dict(ambient=0.2, diffuse=0.8),
        #                                    lightposition=dict(x=0, y=0, z=-1)))
        # fig = go.Figure(data=[*tris_plmesh__, pts_marker])
        # fig.show()

    def to(self, device):
        for attr in dir(self):
            val = getattr(self, attr)
            if type(val) == T.Tensor:
                setattr(self, attr, val.to(device))

        return self

    def numpy(self):
        self.to('cpu')
        for attr in dir(self):
            val = getattr(self, attr)
            if type(val) == T.Tensor:
                setattr(self, attr, val.numpy())

        return self

    # parsing mesh (e.g. adjacency of faces, verts, edges, etc.)
    def _parse_mesh(self, verts, faces_idx, N_repeat_edges=2, N_repeat_verts=9):
        device = verts.device
        meshes = Meshes(verts=[verts], faces=[faces_idx])
        print('parsing mesh topology...')

        # compute faces_to_corres_edges
        faces_to_corres_edges = meshes.faces_packed_to_edges_packed()  # (13776, 3)

        # compute edges_to_corres_faces
        edges_to_corres_faces = T.full((len(meshes.edges_packed()), N_repeat_edges), -1.0).to(device)  # (20664, 2)
        for i in range(len(faces_to_corres_edges)):
            for e in faces_to_corres_edges[i]:
                idx = 0
                while idx < edges_to_corres_faces.shape[1]:
                    if edges_to_corres_faces[e][idx] < 0:
                        edges_to_corres_faces[e][idx] = i
                        break
                    else:
                        idx += 1

        # compute verts_to_corres_faces
        verts_to_corres_faces = T.full((len(verts), N_repeat_verts), -1.0).to(device)  # (6890, 9)
        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                idx = 0
                while idx < verts_to_corres_faces.shape[1]:
                    if verts_to_corres_faces[v][idx] < 0:
                        verts_to_corres_faces[v][idx] = i
                        break
                    else:
                        idx += 1
        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                verts_to_corres_faces[v][verts_to_corres_faces[v] < 0] = verts_to_corres_faces[v][0]

        return faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces

def diagonal_dot(a, b):
    return T.matmul(a * b, T.ones(a.shape[1]).to(a.device))

def points_to_barycentric(triangles, points):
    # Note. swheo: If edge is too small, the computed barycentric might be wrongly set.
    # So multiply it before computing them. 50 is from sanerf's impl
    triangles = triangles
    points = points
    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = points - triangles[:, 0].view((-1, 3))

    dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
    dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
    dot02 = diagonal_dot(edge_vectors[:, 0], w)
    dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
    dot12 = diagonal_dot(edge_vectors[:, 1], w)

    denorm = dot00 * dot11 - dot01 * dot01
    with T.no_grad():
        denorm.clamp_(min=1e-12)
    inverse_denominator = 1.0 / denorm

    barycentric = T.zeros(len(triangles), 3).to(points.device)
    barycentric[:, 2] = (dot00 * dot12 - dot01 *
                         dot02) * inverse_denominator
    barycentric[:, 1] = (dot11 * dot02 - dot01 *
                         dot12) * inverse_denominator
    barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]

    return barycentric

if __name__=="__main__":
    from pytorch3d.utils import ico_sphere

    T.set_printoptions(precision=10)
    test1, test2, test3, test4 = False, True, False, False
    device = T.device('cuda')
    # # Test #1 icosphere projection
    if test1:
        ics_lev = 3
        icsMeshes = ico_sphere(ics_lev)

        verts = icsMeshes.verts_list()[0]
        faces = icsMeshes.faces_list()[0]
        mesh = Mesh(vertices=verts.cpu().tolist(), faces=faces.cpu().tolist())

    # Test #2 smpl
    from dataio.MviewTemporalSMPL import SceneDataset

    dataset = SceneDataset('/ssd3/swheo/db/ZJU_MOCAP/LightStage', subjects=['363'], views=[],
                           num_frame=-878)  # , scale_radius=5) # /ssd2/swheo/db/DTU/scan65
    data = dataset.__getitem__(1)

    mesh = Mesh(file_name='../assets/smpl/smpl/smpl_uv.obj')
    verts = mesh.vertices
    faces = mesh.faces
    cache_path = '../assets/smpl/smpl/' #'../assets/testIcoSphere/'
    dprojfunc = DispersedProjector(cache_path=cache_path, mesh=mesh)
    N_ray = 1
    N_sample = 32
    vertices = data[1]['vertices']
    mesh.update_vertices(vertices)

    if test1 or test2:
        numquery = N_ray * N_sample
        margin = 1.
        bbox = T.stack([mesh.vertices.min(dim=0).values - margin, mesh.vertices.max(dim=0).values + margin])
        # query : [(B), N_rays, N_samples+N_importance, 3]
        query = bbox[0, :] + T.rand((numquery, 3)) * (bbox[-1, :] - bbox[0, :])
        # query = T.from_numpy(np.array([[1.,1.,1.], [-1.,-1.,-1.]], dtype=np.float32))
        query = query.view((1, N_ray, N_sample, -1))
    if test3:
        # Test#3 real case
        c2w = data[1]['c2w']
        intrinsics = data[1]['intrinsics']
        H, W = data[1]['object_mask'].shape
        from utils import rend_util
        margin = 0.1
        bbox = T.stack([data[1]['vertices'].min(dim=0).values - margin, data[1]['vertices'].max(dim=0).values + margin])
        bbox_mask = rend_util.get_2dmask_from_bbox(bbox, intrinsics, c2w, H, W)
        # Test closet surface point search
        from utils.geometry import Mesh, project2closest_face, texcoord2imcoord
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=N_ray, mask=None)
        rays_d = F.normalize(rays_d, dim=-1)
        near, far, valididx = rend_util.near_far_from_bbox(rays_o, rays_d, bbox)
        _t = T.linspace(0, 1, N_sample).float().to(rays_o.device)
        dquery = (near * (1 - _t) + far * _t)
        query = rays_o.unsqueeze(-2) + dquery.unsqueeze(-1) * rays_d.unsqueeze(-2)
        # query = T.Tensor([-0.0078, -0.06779, -0.28070], device=query.device)[None,None,...]
        query = query[None].contiguous()
    #pp_color = np.ones_like(verts)*0.25
    # plotly_viscorres3D(verts, faces, draw_edge=True)

    if test4:
        sigma_global = 3.0 / 2.
        # pretraining way
        local_sigma = T.from_numpy(dataset.local_sigma).unsqueeze(-1).float()
        sample_inds = T.randint(mesh.vertices.shape[0], (N_sample,))
        pts_on = mesh.vertices[sample_inds, :]
        pts_out_loc = pts_on + T.randn_like(pts_on) * local_sigma[sample_inds, :]
        pts_out_glo = T.rand(vertices.shape[0] // 2, 3) * (
                sigma_global * 2) - sigma_global
        pts_out = T.cat([pts_out_loc, pts_out_glo], dim=0)
        query = pts_on.unsqueeze(0)
    query = T.load('/ssd3/swheo/dev/code/neurecon/utils/pts_out.pt', map_location='cpu')

    # query = T.Tensor(, device = bbox.device)
    query = query.to(device)
    mesh = mesh.to(device)
    dprojfunc = dprojfunc.to(device)
    from Disperse_projection_ref import SurfaceAlignedConverter
    # To compare with original implementation
    dprojfunc_comp = SurfaceAlignedConverter(verts=verts.to(device), faces=faces.to(device), device=device, cache_path=cache_path)

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_modules=True) as prof:
        with record_function("dispersed_projeciton_original"):
            out_comp, nearest, nearest_new, barycentric = dprojfunc_comp.xyz_to_xyzch(query.view(1,-1,3), mesh.vertices.unsqueeze(0), xyzc_in=verts.to(device), debug=True) #xyzc

    # print(prof.key_averages().table(sort_by="cuda_time_total"))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_modules=True) as prof:
        with record_function("dispersed_projeciton_ported"):
            vnear_init, st_init, idxs_init = project2closest_face(query, mesh)
            # vnear_i, st_i, idxs_i = project2closest_face(query, mesh, use_cgd=True)
            h, st_, vnear, vnear_, idxs = dprojfunc(query, mesh, vnear=vnear_init, st=st_init, idxs=idxs_init)
            mesh.update_vertices(vertices=verts.to(device))
            scale = mesh.stability_scale
            vnear_tpose = mesh.project_func.B[idxs]/scale + \
                          st_[..., 0].unsqueeze(-1) * mesh.project_func.E0[idxs]/scale + \
                          st_[..., 1].unsqueeze(-1) * mesh.project_func.E1[idxs]/scale
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    pp_color = T.ones_like(query).view(-1,3).cpu().numpy()*0.75
    # plotly_viscorres3D(verts, faces, vnear=vnear_tpose, query=query, pp_color=pp_color, draw_edge=True)
    # plotly_viscorres3D(verts, faces, vnear=out_comp[...,:3], query=query, pp_color=pp_color, draw_edge=True)
    if (not T.all(nearest_new.isclose(vnear_))) or (not T.all(nearest.isclose(vnear))) or (not T.all(out_comp[...,:3].isclose(vnear_tpose))):
        foo = 1
        print(T.norm(vnear_init - nearest))
    foo = 1




