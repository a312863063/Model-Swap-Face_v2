import numpy as np
import torch

def edit(latents, edit_file, strength=6):
    edit_npy = np.load(edit_file)
    delta_padded = strength * torch.FloatTensor(edit_npy).cuda()
    edit_latents = [latent + delta_padded for latent in latents]
    return edit_latents
