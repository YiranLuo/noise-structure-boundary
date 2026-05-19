from Adverse import zero_gradients
import numpy as np
import torch

def clip(current, low_bound, up_bound, device):
    low_bound = torch.FloatTensor(low_bound).to(device)
    up_bound = torch.FloatTensor(up_bound).to(device)
    return torch.max(torch.min(current, up_bound), low_bound)


def deepfool(x_old, net, maxiters, alpha, bounds, weights=[], overshoot=0.002):
    """
    :param image: tabular sample
    :param net: network 
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param bounds: bounds of the datasets with respect to each feature
    :param weights: feature importance vector associated with the dataset at hand
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """    
    input_shape = x_old.detach().cpu().numpy().shape
    x = x_old.detach().clone().requires_grad_(True)
    
    output = net.forward(x)
    orig_pred = output.max(0, keepdim=True)[1]  # get the index of the max log-probability
    origin = orig_pred.clone().detach()
    I = []
    if orig_pred.item() == 0:
        I = [0, 1]
    else:
        I = [1, 0]       
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)    
    k_i = origin.clone()
    loop_i = 0
    while torch.eq(k_i, origin) and loop_i < maxiters:                
        # Origin class
        output[I[0]].backward(retain_graph=True)
        grad_orig = x.grad.detach().cpu().numpy().copy()        
        # Target class
        zero_gradients(x)
        output[I[1]].backward(retain_graph=True)
        cur_grad = x.grad.detach().cpu().numpy().copy()            
        # set new w and new f
        w = cur_grad - grad_orig
        f = (output[I[1]] - output[I[0]]).detach().cpu().numpy()
        pert = abs(f)/np.linalg.norm(w.flatten())    
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)          
        if len(weights) > 0:
            r_i /= np.array(weights)
        # limit huge step
        r_i = alpha * r_i / np.linalg.norm(r_i)            
        r_tot = np.float32(r_tot + r_i)       
        pert_x = x_old + (1 + overshoot) * torch.from_numpy(r_tot).to(x_old.device)
        if len(bounds) > 0:
            pert_x = clip(pert_x, bounds[0], bounds[1], x_old.device)               
        x = pert_x.detach().clone().requires_grad_(True)
        output = net.forward(x)      
        k_i = torch.tensor(np.argmax(output.detach().cpu().numpy().flatten()), device=origin.device)                   
        loop_i += 1
    r_tot = (1+overshoot)*r_tot    
    pert_x = clip(pert_x, bounds[0], bounds[1], x_old.device)
    return int(orig_pred.item()), int(k_i.item()), pert_x.detach().cpu().numpy(), loop_i