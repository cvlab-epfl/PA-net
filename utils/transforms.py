import torch.nn.functional as F
import torch
from utils import affine_3d_grid_generator

def rot_matrix_from_quaternion(params):
    qi, qj, qk = [i[:, 0] for i in torch.split(params, 1, dim=1)]

    s = qi ** 2 + qj ** 2 + qk ** 2

    theta = torch.eye(4, device=params.device)[None].repeat(params.shape[0], 1, 1)

    theta[:, 0, 0] = 1 - 2 * s * (qj ** 2 + qk ** 2)
    theta[:, 1, 1] = 1 - 2 * s * (qi ** 2 + qk ** 2)
    theta[:, 2, 2] = 1 - 2 * s * (qi ** 2 + qj ** 2)

    theta[:, 0, 1] = 2 * s * qi * qj
    theta[:, 0, 2] = 2 * s * qi * qk

    theta[:, 1, 0] = 2 * s * qi * qj
    theta[:, 1, 2] = 2 * s * qj * qk

    theta[:, 2, 0] = 2 * s * qi * qk
    theta[:, 2, 1] = 2 * s * qj * qk
    return theta

def shift(axes):
    theta = torch.eye(4)
    theta[0, 3] = axes[0]
    theta[1, 3] = axes[1]
    theta[2, 3] = axes[2]

    return theta

def transform(theta, x, y):
    theta = theta[:, :3]
    grid = affine_3d_grid_generator.affine_grid(theta, x[None].shape).cuda()
    x = F.grid_sample(x[None], grid, mode='bilinear', padding_mode='zeros')[0]
    y = F.grid_sample(y[None, None].float(), grid, mode='nearest', padding_mode='zeros').long()[0, 0]
    return x, y


