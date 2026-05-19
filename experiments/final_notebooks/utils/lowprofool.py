
import torch
import torch.nn as nn
import numpy as np


def clip_tensor(current, low_bound, up_bound, dev):
    low_bound = torch.FloatTensor(low_bound).to(dev)
    up_bound = torch.FloatTensor(up_bound).to(dev)
    return torch.max(torch.min(current, up_bound), low_bound)


def lowProFool_attack(x, model, weights, bounds, maxiters, alpha, lambda_, device):
    """Run LowProFool on a single sample. Returns (x_adv_numpy, success_flag)."""
    x = x.to(device)
    w = torch.FloatTensor(np.array(weights)).to(device)
    r = torch.FloatTensor(1e-4 * np.ones(x.shape)).to(device)
    r.requires_grad = True

    with torch.no_grad():
        orig_pred = model(x).argmax().cpu().item()

    target_pred = 1 - orig_pred
    target = torch.tensor([0., 1.] if target_pred == 1 else [1., 0.]).to(device)
    bce = nn.BCELoss()

    for _ in range(maxiters):
        if r.grad is not None:
            r.grad.zero_()
        output = model(x + r)
        loss = bce(output, target) + lambda_ * torch.sqrt(torch.sum((w * r) ** 2))
        loss.backward(retain_graph=True)
        with torch.no_grad():
            r_new = r - alpha * r.grad
        r = r_new.clone().detach().requires_grad_(True)

    x_adv = clip_tensor(x + r, bounds[0], bounds[1], device)
    with torch.no_grad():
        adv_pred = model(x_adv).argmax().cpu().item()

    return x_adv.detach().cpu().numpy().flatten(), int(orig_pred != adv_pred)