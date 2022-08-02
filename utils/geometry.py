import torch as T
import numpy as np
import torch.nn.functional as F

from utils.io_util import read_obj

from pytorch3d import _C
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Mesh(object):
    def __init__(self, vertices=None, faces=None, file_name=None, **kwargs):
        assert (np.any(vertices != None) and np.any(faces != None)) or (file_name != None), "at least vertices, faces or path to '.obj' file should be provided"
        if file_name is not None:
            assert file_name.split('.')[-1] == 'obj', "sobj file is supported only"
            obj = read_obj(file_name)
            vertices = obj['v']
            uv = obj['vt']
            faces = obj['f']
            faces_uv = obj['ft']
        else:
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int64)
            uv = np.array(kwargs['uv'], dtype=np.float32) if 'uv' in kwargs.keys() else np.array([])
            faces_uv = np.array(kwargs['faces_uv'], dtype=np.int64) if 'faces_uv' in kwargs.keys() else np.array([])

        self.vertices = T.from_numpy(vertices)
        self.uv = T.from_numpy(uv)
        self.faces = T.from_numpy(faces)
        self.faces_uv = T.from_numpy(faces_uv)

        # Compute face related attributes
        self.update_vertices(self.vertices)

    def update_vertices(self, vertices: T.Tensor):
        self.vertices = vertices
        # Compute face related attributes
        # self.faces_normal, self.faces_area, self.edges = computeFaceNormal(self.vertices, self.faces)
        self.vert_normal, self.faces_normal, self.faces_area, self.edges = computeNormals(self.vertices, self.faces, stability_scale=stability_scale)
        self.triangles = self.vertices[self.faces]
        self.faces_center = self.triangles.mean(dim=-2)
        # self.faces_framerot = computeFrameRot(self.faces_normal)
        self.faces_tanframe = computeTangentFrame(self.faces_normal, self.edges[:,0,:].squeeze()) # Use fixed axis
        ## Compute vars for point to mesh projection
        self.project_func = ProjectMesh2Point(self)

    def get_uv_from_st(self, st:T.tensor, idx):
        """
        get uv coordinate from st values
        """
        # Compute uv related attributes
        tri_uv = self.uv[self.faces_uv]
        edges_uv = T.stack([tri_uv[:,1,:] - tri_uv[:,0,:],
                            tri_uv[:,0,:] - tri_uv[:,2,:],
                            tri_uv[:,2,:] - tri_uv[:,1,:]]).transpose(0,1)
        B = tri_uv[idx,0]
        E0 = edges_uv[idx,0]
        E1 = -edges_uv[idx,1]

        uv = B + st[...,0][..., None] * E0 + st[...,1][..., None] * E1

        return uv

    def to(self, device):
        for attr in dir(self):
            val = getattr(self, attr)
            if type(val) == T.Tensor:
                setattr(self, attr, val.to(device))

        for attr in dir(self.project_func):
            val = getattr(self.project_func, attr)
            if type(val) == T.Tensor:
                setattr(self.project_func, attr, val.to(device))

        return self

    def numpy(self):
        self.to('cpu')
        for attr in dir(self):
            val = getattr(self, attr)
            if type(val) == T.Tensor:
                setattr(self, attr, val.numpy())

        for attr in dir(self.project_func):
            val = getattr(self.project_func, attr)
            if type(val) == T.Tensor:
                setattr(self.project_func, attr, val.numpy())

        return self

    @property
    def device(self):
        return self.vertices.device

pmkeys = ['a','b','c','h', 'delta'] # values that can be precomputed
            # values that should be computed on the fly
            # 'd','e','f','sbar','tbar','g','k','bd','ce','be','ad']
class ProjectMesh2Point(object):
    """
    ref : https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    """
    def __init__(self, mesh :Mesh, stability_scale=1e3):
        self.stability_scale = stability_scale
        # Note. edge is oriented in right-handed direction,
        self.E0 = stability_scale * mesh.edges[:,0].squeeze()
        self.E1 = -stability_scale * mesh.edges[:,1].squeeze()
        self.B = stability_scale * mesh.triangles[:,0].squeeze()
        for key in pmkeys:
            setattr(self, key, [])

        # Values that can be precomputed
        self.a = T.sum(self.E0 * self.E0, dim=-1) # inner product
        self.b = T.sum(self.E0 * self.E1, dim=-1)
        self.c = T.sum(self.E1 * self.E1, dim=-1)
        self.delta = self.a * self.c - self.b**2
        self.h = self.a - 2*self.b + self.c

    def __call__(self, query, index):
        """
        The projected point is computed by
        Pout = B + sout*E0 + tout*E1
        query : Px3 vector
        index : Px1 indices for updateing vars
        """
        # Update projection variables according to input queries
        # variables of "d,e,f,sbar,tbar,g,k,bd,ce,be,ad" are updated
        query = self.stability_scale * query
        # Get vars in indices
        B, E0, E1 = self.B[index,:], self.E0[index,:], self.E1[index,:]
        a, b, c, h = self.a[index], self.b[index], self.c[index], self.h[index]
        delta = self.delta[index]
        D = B - query
        d = T.sum(E0*D, dim=-1)
        e = T.sum(E1*D, dim=-1)
        # f = T.sum(D*D, dim=-1)

        sbar = b * e - c * d
        tbar = b * d - a * e

        bd = b + d
        ce = c + e

        ad = a + d
        be = b + e

        g = ce - bd
        k = ad - be

        # output
        sout = T.zeros_like(sbar, device=sbar.device)
        tout = T.zeros_like(tbar, device=tbar.device)

        # Region classification
        r_conds = T.stack([(sbar+tbar)<=delta, sbar>=0., tbar>=0., bd>ce, d<0, be>ad])

        # Inside triangle
        r_0 = r_conds[0] & r_conds[1] & r_conds[2] # region 0
        sout[r_0] = sbar[r_0]/delta[r_0]
        tout[r_0] = tbar[r_0]/delta[r_0]

        # region 1
        r_1 = ~r_conds[0] & r_conds[1] & r_conds[2]
        sout[r_1] = T.clip(g[r_1]/h[r_1], 0., 1.)
        tout[r_1] = 1 - sout[r_1]

        # region 2
        r_2 = ~r_conds[0] & ~r_conds[1] & r_conds[2]
        r_2a = r_2 & r_conds[3] # region 2-a
        sout[r_2a] = T.clip(g[r_2a] / h[r_2a], 0., 1.)
        tout[r_2a] = 1 - sout[r_2a]
        r_2b = r_2 & ~r_conds[3] # region 2-b
        tout[r_2b] = T.clip(-e[r_2b]/c[r_2b], 0., 1.) # Note. sout=0 in r_2b

        # region 3
        r_3 = r_conds[0] & ~r_conds[1] & r_conds[2]
        tout[r_3] = T.clip(-e[r_3] / c[r_3], 0., 1.)  # Note. sout=0 in r_3

        # region 4
        r_4 = r_conds[0] & ~r_conds[1] & ~r_conds[2]
        r_4a = r_4 & r_conds[4] # region 4-a
        sout[r_4a] = T.clip(-d[r_4a]/a[r_4a], 0., 1.) # Note tout=0 in r_4a
        r_4b = r_4 & ~r_conds[4] # region 4-b
        tout[r_4b] = T.clip(-e[r_4b] / c[r_4b], 0., 1.)  # Note sout=0 in r_4b

        # region 5
        r_5 = r_conds[0] & r_conds[1] & ~r_conds[2]
        sout[r_5] = T.clip(-d[r_5]/a[r_5], 0., 1.) # Note tout=0 in r_5

        # region 6
        r_6 = ~r_conds[0] & r_conds[1] & ~r_conds[2]
        r_6a = r_6 & r_conds[5]
        tout[r_6a] = T.clip(k[r_6a]/h[r_6a], 0., 1.)
        sout[r_6a] = 1 - tout[r_6a]
        r_6b = r_6 & ~r_conds[5]
        tout[r_6b] = T.clip(-d[r_6b]/a[r_6b], 0., 1.) # Note sout=0 in r_6b

        # Sanity check
        # Should be false all
        # r_1 & r_2 & r_3 & r_4 & r_5 & r_6
        # (r_2a & r_2b) | (r_4a & r_4b) | (r_6a & r_6b)

        Pout = B + sout[...,None]*E0 + tout[...,None]*E1

        return Pout/self.stability_scale, (sout, tout)

def computeNormals(v, f):
    tri = v[f]
    edges = T.stack([tri[:,1,:] - tri[:,0,:],
                    tri[:,0,:] - tri[:,2,:],
                    tri[:,2,:] - tri[:,1,:]], dim=-1).transpose(-2,-1)
    # |p1-p0| x |p2-p0|
    v_cross = T.cross(edges[:,0,:], -edges[:,1,:])
    v_cross_norm = T.sqrt(T.sum(v_cross**2, dim=1))

    f_norm = v_cross/v_cross_norm[...,None]

    # Uniform normals
    v_norm = T.zeros_like(v)
    v_norm = v_norm.index_add(0, f.reshape(-1), v_cross.repeat(1, 3).reshape(-1, 3))
    v_norm = v_norm/T.linalg.norm(v_norm, dim=-1).unsqueeze(-1)

    return v_norm, f_norm, v_cross_norm, edges

def computeTangentFrame(vn, b_X=None):
    """
    Modified from https://github.com/nmwsharp/diffuion-net/src/geometry.py#L151
    """
    #todo rather than b_cand1, fix edge1 and compute other
    nverts = vn.shape[0]
    dtype = vn.dtype
    device = vn.device

    if b_X == None:
        ## Find an orthogonal basis
        b_cand1 = T.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(nverts, -1)
        b_cand2 = T.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(nverts, -1)

        b_X = T.where((T.abs(T.sum(vn*b_cand1, dim=-1)) #dot
                       <0.9).unsqueeze(-1), b_cand1, b_cand2)
        b_X = b_X - vn*T.sum(b_X*vn, dim=-1).unsqueeze(-1) #project_to_tangent

    # Note. F.normalize gurantees frames to lie in SO(3) group
    b_X = F.normalize(b_X) #b_X / (T.norm(b_X, dim=-1)+1e-6).unsqueeze(-1) #normalize
    b_Y = T.cross(vn, b_X, dim=-1)
    # todo orthonormality check
    frames = T.stack((b_X, b_Y, vn), dim=-2)

    if T.any(T.isnan(frames)):
        raise ValueError("Nan coordinate frame! Must be degenerate")

    return frames

# def computeFrameRot(normal):
#     """
#     Modified from diffusionnet, https://github.com/nmwsharp/diffusion-net
#
#     """
#     # R @ normal = z
#     z = T.tensor([0.,0.,1.], dtype=T.float32, device=normal.device).expand_as(normal)
#     a = T.cross(normal, z)
#     a = a / T.linalg.norm(a, dim=-1).unsqueeze(-1)
#     b = T.cross(normal, a)
#     b = b / T.linalg.norm(b, dim=-1).unsqueeze(-1)
#     R = T.stack([normal, a, b], dim=-2)
#
#     # degeneracy check
#     assert T.linalg.norm(a, dim=-1).min()>0, 'computeFrameRot has error, need degeneracy check'
#
#     return R

    # degeneracy check
    assert T.linalg.norm(a, dim=-1).min()>0, 'computeFrameRot has error, need degeneracy check'

    return R

def project2closest_face(query, mesh: Mesh, stability_scale=1e3):
    """
    query : [(B), N_rays, N_samples+N_importance, 3]
    mesh : Mesh class.
    ref : https://github.com/facebookresearch/pytorch3d/issues/1045
    """
    if len(query.shape) == 3:
        N_ray = 1
        B, N_pts, _ = query.shape
    elif len(query.shape) == 4:
        B, N_ray, N_pts, _ = query.shape
    else:
        N_ray, B = 1, 1
        N_pts, _ = query.shape
    # tris = T.cat(B * [mesh.faces_center[None,...]])
    tris = mesh.triangles

    # Batch to shape fit to point_face_dist_foward
    vquery = query.view(-1, 3)

    # point to tri distance
    dists, idxs = _C.point_face_dist_forward(
        vquery*stability_scale, T.zeros((1,), device=tris.device, dtype=T.int64),
        tris*stability_scale, T.zeros((1,), device=tris.device, dtype=T.int64), vquery.size()[0]
    )

    vnear, st = mesh.project_func(vquery, idxs)

    st_ = T.cat([st[0].view((B, N_ray, N_pts, -1)), st[1].view((B, N_ray, N_pts, -1))], dim=-1)
    vnear = vnear.view((B, N_ray, N_pts, -1))
    idxs = idxs.view((B, N_ray, N_pts))
    return vnear, st_, idxs

def texcoord2imcoord(vt, height, width):
    angle = np.pi / 2.
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], 'float32')
    R = T.from_numpy(R).to(vt.device)
    vt_new = T.matmul(vt - .5, R.t()) + .5
    u, v = T.split(vt_new, (1, 1), -1)
    uv = T.cat(((width - 1) * v, (height - 1) * u), -1)
    return uv

if __name__=="__main__":
    from pytorch3d.utils import ico_sphere
    from pytorch3d.io import IO
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from zju_smpl.body_model import SMPLlayer
    # smpl_dir = '../assets/smpl/smpl/smpl_uv.obj'
    # smpl = SMPLlayer('../assets/smpl/smpl', model_type='smpl', gender='neutral')
    # verts = smpl.v_template.numpy()
    # faces = smpl.faces
    # faces = mesh_.faces_list()[0].numpy()
    # mesh = Mesh(vertices=smpl.v_template, faces=smpl.faces)
    # mesh = IO().load_mesh(smpl_dir)
    # mesh_ = ico_sphere(1)
    # verts = mesh_.verts_list()[0].numpy()
    # faces = mesh_.faces_list()[0].numpy()
    # mesh = Mesh(vertices=verts, faces=faces)

    # Green for smpl mesh
    # color = T.tensor([0., 1., 0.]).view(1, 1, 3).expand(-1, smpl.nVertices, -1)
    # tex = TexturesVertex(color)
    # mesh = Meshes(verts=[smpl.v_template], faces=[smpl.faces_tensor], textures=tex)

    # get triangles
    # verts = mesh.verts_list()[0]
    # faces = mesh.faces_list()[0]
    # tris = verts[faces]
    # mesh = Mesh(vertices=smpl.v_template, faces=smpl.faces)
    mesh = Mesh(file_name='../assets/smpl/smpl/smpl_uv.obj')
    N_ray = 128
    N_sample = 128
    numquery = N_ray * N_sample
    bbox = T.stack([mesh.vertices.min(dim=0).values, mesh.vertices.max(dim=0).values])
    # query : [(B), N_rays, N_samples+N_importance, 3]
    query = bbox[0,:] + T.rand((numquery, 3))*(bbox[-1,:] - bbox[0,:])
    # query = T.from_numpy(np.array([[1.,1.,1.], [-1.,-1.,-1.]], dtype=np.float32))
    query = query.view((1, N_ray, N_sample, -1))

    device = T.device('cuda:7')

    from torch.profiler import profile, record_function, ProfilerActivity
    query = query.to(device)
    mesh = mesh.to(device)
    vnear, st, idxs = project2closest_face(query, mesh)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_modules=True) as prof:
        with record_function("point_to_mesh_projection"):
            vnear, st, idxs = project2closest_face(query, mesh)

    print(prof.key_averages().table(sort_by="cuda_time_total"))

    uv = mesh.get_uv_from_st(st, idxs)

    # Visualize
    idx = idxs.cpu().numpy()
    pp = vnear.cpu().numpy()
    qq = query.view(-1, 3).cpu().numpy()
    x,y,z, = mesh.vertices.cpu().chunk(3, -1)
    faces = mesh.faces.cpu()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")
    # ax.plot_trisurf(x.numpy()[faces[idxs],0], y.numpy()[faces[idxs],0], z.numpy()[faces[idxs],0], triangles=np.array([0,1,2], dtype=np.int32), alpha=0.5, edgecolor=[0,0,0])
    ax.plot_trisurf(x.numpy()[:,0], y.numpy()[:,0], z.numpy()[:,0],
                    triangles=faces, alpha=0.5, edgecolor=[0, 0, 0])
    # ax.scatter3D(x[:, 0].numpy(), y[:, 0].numpy(), z[:, 0].numpy(), s=10, c="green")
    ax.scatter3D(pp[:, 0], pp[:, 1], pp[:, 2], s=20, c="red")
    ax.scatter3D(qq[:, 0], qq[:, 1], qq[:, 2], s=20, c="red")
    for qqq, ppp in zip(qq, pp):
        ax.plot([qqq[0], ppp[0]], [qqq[1], ppp[1]], [qqq[2], ppp[2]], c="blue")
    foo = 1
