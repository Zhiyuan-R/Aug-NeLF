import torch

def get_persp_intrinsic(H, W, focal, ps=1., us=1., device=torch.device('cpu')):

    return torch.tensor([
        [focal*us/ps, 0, W/2],
        [0, focal*us/ps, H/2],
        [0, 0, 1]
    ], device=device)
    
def get_persp_rays(H, W, K, c2w, z_dir=-1.):

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=c2w.device), torch.linspace(0, H-1, H, device=c2w.device))
    i, j = i.t(), j.t()
    
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    return torch.stack([rays_o, rays_d], 0)
