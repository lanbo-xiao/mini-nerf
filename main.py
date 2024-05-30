import torch
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
from torch import nn
import torch.nn.functional as F
from load_data import load_data_lego
from nerf_model import Nerf


#Position Encoding
def PE(x, L):
    pe = []
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2. ** i * x))
    return torch.cat(pe, -1)


#return the rays_o and rays_d
def sample_rays(H, W, f, c2w):
    #every pixel
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    #pixel in camera coordinate
    #dirs.shape=(W,H,3)
    dirs = np.stack([(i - W * .5 + .5) / f, -(j - H * .5 + .5) / f, -np.ones_like(i)], -1)

    #rays_o is the rays‘ origin
    #rays_d is the ray's direction
    #pretty exquisite
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    rays_o = torch.from_numpy(rays_o).float().to("cuda")
    rays_d = torch.from_numpy(rays_d).float().to("cuda")
    return rays_o, rays_d


def uniform_sample_point(tn, tf, N_samples):
    #Divide the interval [tn, tf] into N_samples segments
    #and use uniform sampling to obtain sampling points in each segment.
    k = torch.rand([N_samples]) / float(N_samples)
    pt_value = torch.linspace(0.0, 1.0, N_samples + 1)[:-1]
    pt_value += k
    return tn + (tf - tn) * pt_value


def volumn_render(rays_o, rays_d, Nerf, N_sample):
    z_vals = uniform_sample_point(2.0, 6.0, N_sample)
    #pts => tensor(N_rand, N_sample, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    # Position Encodeing pts_PE => tensor(N_rand, N_sample, 60)
    pts_PE = PE(pts, L=10)

    dir = F.normalize(rays_d.unsqueeze(-2).expand_as(pts), p=2, dim=-1)
    # dir_PE => tensor(N_rand, N_sample, 24)
    dir_PE = PE(dir, L=4)

    rgb, sigma = Nerf(pts_PE, dir_PE)
    sigma = sigma.squeeze()

    #volumn_render
    delta = z_vals[1:] - z_vals[:-1]
    INF = torch.ones(delta[..., :1].shape).fill_(1e10)
    delta = torch.cat([delta, INF], -1)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)
    alpha = 1. - torch.exp(-sigma * delta)
    ones = torch.ones(alpha[..., :1].shape, device=device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]

    return rgb, weights


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load data
    imgs, poses, [H, W, focal] = load_data_lego()
    print("data loaded")
    print("image.shape:", imgs.shape)
    print("H,W,focal:", H, W, focal)

    #create model
    #nerf = Nerf().to(device)
    nerf = torch.load("nerf.model")
    optimizer = torch.optim.Adam(nerf.parameters(), lr=1e-4)
    print("model created")

    #Training and config
    epochs = 1000
    loss = nn.MSELoss()
    #how many points needed to be sampled along one ray
    N_sample = 64
    #how many rays
    N_rand = 1024
    for epoch in trange(epochs):
        # random select one image
        random_idx = np.random.randint(0, imgs.shape[0])
        image = imgs[random_idx]
        pose = poses[random_idx]

        #get rays
        rays_o, rays_d = sample_rays(H, W, focal, c2w=pose)

        #random select N_rand rays
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                             -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long().to("cpu")  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        target_s = image[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        target_s = torch.tensor(target_s, dtype=torch.float32)

        #volumn render
        rgb, weights = volumn_render(rays_o=rays_o, rays_d=rays_d, Nerf=nerf, N_sample=N_sample)
        pred_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
        l = loss(pred_rgb, target_s)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print(l)
    torch.save(nerf, "nerf.model")

    #visualization
    with torch.no_grad():
        show_image = []
        for i in range(800):
            pose = poses[10]

            rays_o, rays_d = sample_rays(H, W, focal, c2w=pose)

            rays_o = rays_o[i]
            rays_d = rays_d[i]

            rgb, weights = volumn_render(rays_o=rays_o, rays_d=rays_d, Nerf=nerf, N_sample=N_sample)
            pred_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
            show_image.append(pred_rgb)
        show_image = torch.stack(show_image, dim=0)
        plt.imshow(show_image.cpu().detach().numpy())
        plt.show()
