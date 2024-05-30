import os
import numpy as np
import torch
import json
import imageio


# # z = z + t
# trans_t = lambda t: torch.Tensor([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, t],
#     [0, 0, 0, 1]]).float()
#
# # rotate y and z Coordinate axis
# rot_phi = lambda phi: torch.Tensor([
#     [1, 0, 0, 0],
#     [0, np.cos(phi), -np.sin(phi), 0],
#     [0, np.sin(phi), np.cos(phi), 0],
#     [0, 0, 0, 1]]).float()
#
# # rotate x and z Coordinate axis
# rot_theta = lambda th: torch.Tensor([
#     [np.cos(th), 0, -np.sin(th), 0],
#     [0, 1, 0, 0],
#     [np.sin(th), 0, np.cos(th), 0],
#     [0, 0, 0, 1]]).float()
#
#
# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi / 180 * np.pi) @ c2w
#     c2w = rot_theta(theta / 180 * np.pi) @ c2w
#     c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
#     return c2w

def load_data_lego():
    try:
        with open('lego/transforms_train.json', 'r') as f:
            meta = json.load(f)
    except:
        print("load data error")

    imgs = []
    poses = []
    for frame in meta['frames']:
        fname = os.path.join("lego/"+frame['file_path'] + '.png')
        image = imageio.imread(fname)
        imgs.append(image)
        poses.append(np.array(frame['transform_matrix']))

    #normmlization
    imgs = np.array(imgs) / 255
    poses = np.array(poses)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    #white background: RGBA -> RGB
    imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])

    # view direction when we test
    # This is actually 40 camera poses,
    # which are used to generate a camera trajectory for synthesizing a new perspective
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
    #                            dim=0)
    return imgs, poses, [int(H), int(W), focal]