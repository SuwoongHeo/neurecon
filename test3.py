import torch
import matplotlib

from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.io import load_ply
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturesVertex, \
    FoVPerspectiveCameras, RasterizationSettings, PointLights, HardPhongShader, \
    DirectionalLights
from pytorch3d.structures import Meshes
from dataio.MviewTemporalSMPL import SceneDataset

class BlendParams:
    def __init__(self, sigma=None, gamma=None, background_color=None):
        self.sigma: float = 1e-4 if sigma is None else sigma
        self.gamma: float = 1e-4 if gamma is None else gamma
        self.background_color = (0.0, 0.0, 0.0) if background_color is None else background_color

device = torch.device("cuda:0")
torch.cuda.set_device(device)
matplotlib.use('TkAgg')
dataset = SceneDataset('/ssd2/swheo/db/ZJU_MOCAP/LightStage', subjects=['363'], views=[], start_frame=878, end_frame=-1,
                       num_frame=1, scale_radius=3.0) # /ssd2/swheo/db/DTU/scan65

meshdir = '/ssd2/swheo/dev/code/neurecon/logs/garmnerf_masksegm_exp_p_b_1/run/test/meshes/000878.ply'
data = dataset.__getitem__(0)
c2w = torch.linalg.inv(data[1]['c2w'])[:3].to(device)
R = c2w[None,:,:3]
tvec = c2w[None,:,3]
intrinsics = data[1]['intrinsics'][None, :3,:3].to(device)
H, W = data[1]['object_mask'].shape

camera = cameras_from_opencv_projection(R=R, tvec=tvec, camera_matrix=intrinsics, image_size=torch.as_tensor([[H, W]], device=device)).to(device)

verts, faces = load_ply(meshdir)

verts_rgb = torch.ones_like(verts)[None]
textures = TexturesVertex(verts_features=verts_rgb.to(device))

mesh = Meshes(verts = [verts.to(device)], faces=[faces.to(device)], textures=textures)

blend_params = BlendParams()
raster_settings = RasterizationSettings(
    image_size=(H,W),
    blur_radius=0.0,
    faces_per_pixel=1,
)

# lights = PointLights(device=device, location=tvec, diffuse_color=((0.5,0.5,0.5),), ambient_color=((0.3,0.3,0.3),))
# lights = DirectionalLights(device=device, direction=((0, -1, 1),), diffuse_color=((0.5,0.5,0.5),), ambient_color=((0.3,0.3,0.3),))
# lights = DirectionalLights(device=device, direction=torch.nn.functional.normalize(tvec), diffuse_color=((0.5,0.5,0.5),), ambient_color=((0.3,0.3,0.3),))
lights = DirectionalLights(device=device, direction=(-R.transpose(-1,-2)@tvec[...,None])[...,0], diffuse_color=((0.5,0.5,0.5),), ambient_color=((0.3,0.3,0.3),))

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=camera, lights=lights, blend_params=blend_params)
)

img = renderer(mesh)
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(data[2]['rgb'].reshape((H,W,-1)).cpu().numpy())
plt.figure()
plt.imshow(img[0,...,:3].cpu().numpy())
plt.show()
foo = 1