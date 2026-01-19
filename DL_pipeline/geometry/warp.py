# geometry/warp.py
import torch
import torch.nn.functional as F

def theta_to_H(theta):
    B = theta.shape[0]
    ones = torch.ones(B, 1, device=theta.device)
    return torch.cat([theta, ones], dim=1).view(B, 3, 3)

def warp(img, H):
    B, C, Hh, Wh = img.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, Hh, device=img.device),
        torch.linspace(-1, 1, Wh, device=img.device),
        indexing="ij"
    )
    grid = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    grid = grid.view(-1, 3).T.unsqueeze(0).repeat(B, 1, 1)

    warped = H @ grid
    warped = warped / (warped[:, 2:3] + 1e-8)

    x = warped[:, 0].view(B, Hh, Wh)
    y = warped[:, 1].view(B, Hh, Wh)

    return F.grid_sample(img, torch.stack([x, y], dim=-1), align_corners=True)
