import numpy as np
import torch


def read_sample(args, path):
    dic = np.load(path)
    x, label = dic['x'], dic['y']
    x = torch.from_numpy(x).float()
    args.label = label

    x = move_to_gpu(x)
    return x


def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1])
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
    return kl_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.xavier_normal_(m.weight)
        elif m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif classname.find('Norm') != -1:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.xavier_normal_(m.weight)
        elif m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
