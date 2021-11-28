import torch
import argparse
from models.psp import pSp
from models.encoders.psp_encoders import Encoder4Editing


def setup_model(args):
    net = pSp(args)
    net.eval()
    net = net.to(args.device)
    return net, net.opts


def load_e4e_standalone(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = argparse.Namespace(**ckpt['opts'])
    e4e = Encoder4Editing(50, 'ir_se', opts)
    e4e_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    e4e.load_state_dict(e4e_dict)
    e4e.eval()
    e4e = e4e.to(device)
    latent_avg = ckpt['latent_avg'].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

    e4e.register_forward_hook(add_latent_avg)
    return e4e
