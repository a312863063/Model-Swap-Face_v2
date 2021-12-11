import matplotlib
matplotlib.use('Agg')
import argparse
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from models.face_parser import BiSeNet
from configs.paths_config import model_paths


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, args):
        super(pSp, self).__init__()

        # Load dict
        encoder_dict = torch.load(args.checkpoint_path_E, map_location='cpu')  # {'state_dict', 'opts'}
        decoder_dict = torch.load(args.checkpoint_path_D, map_location='cpu')  # {'g_ema', 'latent_avg'}
        parser_dict = torch.load(args.checkpoint_path_P, map_location='cpu')
        opts = encoder_dict['opts']
        opts['checkpoint_path'] = args.checkpoint_path_E
        opts['device'] = args.device
        self.opts = argparse.Namespace(**opts)
        
        # Define architecture according to opts
        self.encoder = self.set_encoder(self.opts)
        self.decoder = Generator(self.opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.parser = BiSeNet(n_classes=19)
        
        # Load weights
        self.encoder.load_state_dict(encoder_dict['state_dict'], strict=True)
        self.decoder.load_state_dict(decoder_dict['g_ema'], strict=True)
        self.latent_avg = decoder_dict['latent_avg'].to(self.opts.device)
        self.parser.load_state_dict(parser_dict, strict=True)

    def set_encoder(self, opts):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', opts)
        else:
            raise Exception('{} is not a valid encoders'.format(opts.encoder_type))
        return encoder

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
