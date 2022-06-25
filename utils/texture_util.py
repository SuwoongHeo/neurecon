import os
import numpy as np
import torch as T
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.cameras import FoVOrthographicCameras
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    TexturesUV,
    TexturesVertex,
    MeshRenderer,
)
from pytorch3d import _C

class BlendParams:
    def __init__(self, sigma=None, gamma=None, background_color=None):
        self.sigma: float = 1e-4 if sigma is None else sigma
        self.gamma: float = 1e-4 if gamma is None else gamma
        self.background_color = (0.0, 0.0, 0.0) if background_color is None else background_color

class SimpleShader(T.nn.Module):
    # https://github.com/facebookresearch/pytorch3d/issues/84
    def __init__(self, sigma=None, gamma=None, background_color=None):
        super().__init__()
        self.blend_params = BlendParams(sigma, gamma, background_color)

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        # images (N, H, W, 3) RGBA
        # barycoord (N, H, W, #face, 3)
        # faceinds (N, H, W, 1) Face index for each pixel
        return images, fragments.bary_coords, fragments.pix_to_face  # (N, H, W, 3) RGBA image

class VertColor2UVMapper(object):
    def __init__(self, vt:T.Tensor, ft:T.Tensor, tex_mask:T.Tensor, find_border:bool=False, device=T.device('cpu')):
        super().__init__()
        self.tex_mask = tex_mask.to(device)
        self.tex_size = tex_mask.shape[1]
        self.device = device
        ortho_cam = FoVOrthographicCameras(znear=-1., zfar=1., max_y=1., min_y=0., max_x=1., min_x=0.)
        ras_settings = RasterizationSettings(image_size=self.tex_size, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=ortho_cam, raster_settings=ras_settings)
        self.renderer = MeshRenderer(rasterizer, SimpleShader(background_color=T.zeros(3))).to(self.device)
        self.bcoord, self.faceinds = self.computeUVBcoords(vt.to(device), ft.to(device), self.tex_mask, find_border=find_border)
        self.to(device)

    def __call__(self, vcolor, f, bcoord=None, faceinds=None):
        bcoord = self.bcoord if bcoord==None else bcoord
        faceinds = self.faceinds if faceinds==None else faceinds

        nch = vcolor.shape[1]

        outmap = T.zeros((self.tex_size, self.tex_size, nch), dtype=vcolor.dtype, device=vcolor.device)
        tric = vcolor[f]
        vc0 = tric[:,0,:]
        vc1 = tric[:,1,:]
        vc2 = tric[:,2,:]
        mask = self.tex_mask>0
        vcolor_interp = bcoord[..., 0][mask][..., None] * vc0[faceinds[..., 0][mask]] + \
                bcoord[..., 1][mask][..., None] * vc1[faceinds[..., 0][mask]] + \
                bcoord[..., 2][mask][..., None] * vc2[faceinds[..., 0][mask]]
        outmap[mask[0, ..., None].tile(3)] = vcolor_interp.reshape(-1)
        return outmap

    def computeUVBcoords(self, vt:T.Tensor, ft:T.Tensor, tex_mask:T.Tensor, find_border:bool=False):
        """
        vt : NumUV x 3 , T.float32
        ft : NumFace x 3 , T.int64
        tex_mask : Nch x H, W, T.float32
        find_border : If True,
            Find barycentric coordinate for pixels on tex_mask that are not having instersection with projected (vt,ft)
            This process alleviate the texture seam problem when it comes to rendering
        """
        vt3d = T.dstack((1 - vt[:, 0], vt[:, 1], T.ones(vt.shape[0], device=self.device)))
        texture = TexturesUV(tex_mask[..., None], faces_uvs=ft[None], verts_uvs=vt[None])
        meshes = Meshes(vt3d, ft[None], textures=texture)
        _, bcoord, faceinds = self.renderer(meshes)
        mask_diff = tex_mask - (faceinds[...,0]>=0).type(tex_mask.dtype)
        test_border = mask_diff !=0
        if find_border and T.any(test_border):
            pquery_d = T.where(test_border)
            # Flip to match vt's coordinate system
            # pquery = T.dstack((1.- pquery_d[2].float()/(self.tex_size-1),
            #                    1.- pquery_d[1].float()/(self.tex_size-1),
            #                    T.ones_like(pquery_d[0], device=self.device).float()))[0]
            pquery = T.dstack(((self.tex_size - 1) - pquery_d[2].float() - 0.5,
                            (self.tex_size - 1) - pquery_d[1].float() - 0.5,
                            T.ones_like(pquery_d[0], device=self.device).float()))[0]
            tris = vt3d[0,ft]
            # Note. pytorch3d is not stable with small number.
            tris[..., 0] *= self.tex_size
            tris[..., 1] *= self.tex_size
            startind = T.zeros((1,), device=self.device, dtype=T.int64)
            dists, idxs = _C.point_face_dist_forward(
                pquery, startind, tris, startind, pquery.size()[0]
            )

            bcoords_bd = barycentric_coordinates_(pquery[:,:2],
                                     tris[idxs,0,:2], #v0
                                     tris[idxs,1,:2], #v1
                                     tris[idxs,2,:2]) #v2

            faceinds[test_border] = idxs.unsqueeze(-1)
            bcoord[..., 0][mask_diff != 0] = bcoords_bd[0].unsqueeze(-1)
            bcoord[..., 1][mask_diff != 0] = bcoords_bd[1].unsqueeze(-1)
            bcoord[..., 2][mask_diff != 0] = bcoords_bd[2].unsqueeze(-1)

        return bcoord.squeeze(-2), faceinds

    def to(self, device):
        self.device = device
        for attr in dir(self):
            val = getattr(self, attr)
            if type(val) == T.Tensor:
                setattr(self, attr, val.to(device))

        self.renderer.to(device)

        return self

    def numpy(self):
        self.to('cpu')
        for attr in dir(self):
            val = getattr(self, attr)
            if type(val) == T.Tensor:
                setattr(self, attr, val.numpy())

        return self

def barycentric_coordinates_(p, v0, v1, v2):
    """
    https://github.com/facebookresearch/pytorch3d/blob/34bbb3ad322e1b898044ff763cfaf9c6b7d5a313/pytorch3d/renderer/mesh/rasterize_meshes.py#L698
    p : Nx2 uv coordinates in [0,1]x[0,1]
    v0,v1,v2 : Mx2 uv coordinate of each triangle vertices
    """
    def edge_function(p, v0, v1):
        # Cross-product between A and B where,
        # A = v1 - v0
        # B = p - v0
        return (p[:,0] - v0[:,0]) * (v1[:,1] - v0[:,1]) - (p[:,1] - v0[:,1]) * (v1[:,0] - v0[:,0])
    area = edge_function(v2, v0, v1) + 1e-8 # 2 x face area.
    w0 = edge_function(p, v1, v2) / area
    w1 = edge_function(p, v2, v0) / area
    w2 = edge_function(p, v0, v1) / area
    return (w0, w1, w2)

if __name__=="__main__":
    device = T.device("cuda:5")
    from zju_smpl.body_model import SMPLlayer
    from utils.geometry import Mesh, project2closest_face, texcoord2imcoord
    from utils.io_util import load_rgb
    mesh = Mesh(file_name='../assets/smpl/smpl/smpl_uv.obj').to(device)
    uvmap = T.from_numpy(load_rgb('../assets/smpl/smpl/smpl_uv.png', downscale=4, anti_aliasing=False, order=0)).to(device)
    uvmask = uvmap[-1]
    uvmap = uvmap[:-1] * uvmask.unsqueeze(0)

    Renderer = VertColor2UVMapper(mesh.uv, mesh.faces_uv, uvmask[None], find_border=True).to(device)
    tex = Renderer(mesh.vertices, mesh.faces)

    foo = 1

    import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect("auto")
    # x, y, z, = mesh.vertices.cpu().chunk(3, -1)
    # ax.plot_trisurf(x.numpy()[:, 0], y.numpy()[:, 0], z.numpy()[:, 0],
    #                 triangles=mesh.faces.cpu().numpy(), alpha=0.5, edgecolor=[0, 0, 0], color=mesh.vertices.cpu().numpy()/mesh.vertices.cpu().max())


    plt.show()