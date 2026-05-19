# fgsm_attack.py
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import torch
import numpy as np

def fgsm_attack(x_sample, model, fgsm_attack_obj, device):
   
    # Get original prediction
    x_t = torch.FloatTensor(x_sample.reshape(1, -1)).to(device)
    orig_pred = model(x_t).argmax(dim=1).cpu().numpy()[0]
    
    # Generate adversarial example
    x_adv = fgsm_attack_obj.generate(x=x_sample.reshape(1, -1))[0]
    
    # Get adversarial prediction
    x_adv_t = torch.FloatTensor(x_adv.reshape(1, -1)).to(device)
    adv_pred = model(x_adv_t).argmax(dim=1).cpu().numpy()[0]
    
    return int(orig_pred), int(adv_pred), x_adv


def create_fgsm(model, bounds_min, bounds_max, eps=0.0001):

    wrapped_model = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        input_shape=(bounds_min.shape[0],),
        nb_classes=2,
        clip_values=(bounds_min, bounds_max)
    )
    
    fgsm = FastGradientMethod(estimator=wrapped_model, norm=np.inf, eps=eps)
    
    return wrapped_model, fgsm