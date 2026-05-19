from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import MomentumIterativeMethod, ProjectedGradientDescent
import torch
import numpy as np


def pgd_attack(x_sample, model, pgd_attack_obj, device):
    # Get original prediction
    x_t = torch.FloatTensor(x_sample.reshape(1, -1)).to(device)
    orig_pred = model(x_t).argmax(dim=1).cpu().numpy()[0]
    
    # Generate adversarial example
    x_adv = pgd_attack_obj.generate(x=x_sample.reshape(1, -1))[0]
    
    # Get adversarial prediction
    x_adv_t = torch.FloatTensor(x_adv.reshape(1, -1)).to(device)
    adv_pred = model(x_adv_t).argmax(dim=1).cpu().numpy()[0]
    
    return int(orig_pred), int(adv_pred), x_adv


def create_pgd(model, bounds_min, bounds_max, max_iter, eps, eps_step=0.01):
    wrapped_model = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        input_shape=(bounds_min.shape[0],),
        nb_classes=2,
        clip_values=(bounds_min, bounds_max)
    )
    
    pgd = ProjectedGradientDescent(
        estimator=wrapped_model,
        norm=np.inf,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter
    )
    
    return wrapped_model, pgd
