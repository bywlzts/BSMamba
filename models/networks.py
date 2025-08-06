import torch
import models.bsmamba.BSMamba as BSMamba

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'BSMamba':
        netG = BSMamba.BSMamba(in_channels=3, dim=16, d_state=16, num_blocks=4)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

