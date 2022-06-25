from models.base import ImplicitSurface, NerfppNetwithAutoExpo
from utils.mesh_util import extract_mesh, extract_mesh_nerfpp
from utils.dist_util import init_env

import torch

def main_function(args):
    device = torch.device(f'cuda:{4}')
    # device = torch.device(f'cuda:{args.gpuid}')
    torch.cuda.set_device(device)

    N = args.N
    s = args.volume_size
    if args.isnerfpp:
        ## mesh from density, nerfpp case
        nerfpp = NerfppNetwithAutoExpo().to(device)
        assert args.load_pt is not None, "Need trained nerfpp model, specify --load_pt"
        state_dict = torch.load(args.load_pt, map_location=device)
        # state_dict = torch.load('../logs/nerfpp_jungwoo_exp_b_1_i/ckpts/latest.pt')
        finenet_state_dict = {k.replace('finenet.', ''): v for k, v in state_dict['model'].items() if 'finenet.' in k}
        nerfpp.load_state_dict(finenet_state_dict)
        extract_mesh_nerfpp(nerfpp, s, N=N, filepath=args.out, show_progress=True, chunk=args.chunk)
    else:
        implicit_surface = ImplicitSurface(radius_init=args.init_r).to(device)
        if args.load_pt is not None:
            # --------- if load statedict
            # state_dict = torch.load("/home/PJLAB/guojianfei/latest.pt")
            # state_dict = torch.load("./dev_test/37/latest.pt")
            state_dict = torch.load(args.load_pt, map_location=device)
            imp_surface_state_dict = {k.replace('implicit_surface.',''):v for k, v in state_dict['model'].items() if 'implicit_surface.' in k}
            imp_surface_state_dict['obj_bounding_size'] = torch.tensor([1.0]).cuda()
            implicit_surface.load_state_dict(imp_surface_state_dict)
        if args.out is None:
            from datetime import datetime
            dt = datetime.now()
            args.out = 'surface_' + dt.strftime("%Y%m%d%H%M%S") + '.ply'
        extract_mesh(implicit_surface, s, N=N, filepath=args.out, show_progress=True, chunk=args.chunk)
    foo = 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help='output ply file name')
    parser.add_argument('--N', type=int, default=512, help='resolution of the marching cube algo')
    parser.add_argument('--volume_size', type=float, default=2., help='voxel size to run marching cube')
    parser.add_argument("--load_pt", type=str, default=None, help='the trained model checkpoint .pt file')
    parser.add_argument("--isnerfpp", action='store_true', default=False, help='whether to use implicit rep. or nerfpp rep.')
    parser.add_argument("--gpuid", type=str, default='0', help='gpuid to use')
    parser.add_argument("--chunk", type=int, default=16*1024, help='net chunk when querying the network. change for smaller GPU memory.')
    parser.add_argument("--init_r", type=float, default=1.0, help='Optional. The init radius of the implicit surface.')
    args = parser.parse_args()
    
    main_function(args)