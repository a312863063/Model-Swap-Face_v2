import argparse
import torch
import numpy as np
import sys
import os
import dlib
sys.path.append(".")
sys.path.append("..")
from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image
import cv2
import shutil
from editings.easy_edit import edit


def main(args):
    # Prepare and set-up model
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    torch.nn.Module.dump_patches = True
    torch.multiprocessing.set_start_method('spawn')

    net, opts = setup_model(args)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts, parse_net=net.parser)

    # Get dlatents
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    latent_codes = get_all_results(net, data_loader, args.n_sample, is_cars=is_cars, device=args.device)
    torch.save(latent_codes, latents_file_path)

    # Edit dlatents
    #latent_codes = edit(latent_codes, 'editings/latent_directions/emotion_fear.npy', strength=6)

    # Generate new faces and merge back
    projected_images = generate_inversions(args, generator, latent_codes, is_cars=is_cars)


def setup_data_loader(args, opts, parse_net=None):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=run_alignment,
                                    opts=opts,
                                    parse_net=parse_net)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_results(net, data_loader, n_images=None, is_cars=False, device='cuda', load_latents=None):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            img, x = batch
            if not load_latents:
                inputs = x.to(device).float()
                latents = get_latents(net, inputs, is_cars)
                all_latents.append(latents)
            i += 1
    if not load_latents:
        all_latents = torch.cat(all_latents)
    else:
        all_latents = torch.load(load_latents)
        all_latents = torch.cat(all_latents).to(device)
    return all_latents


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)

@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    print('Saving inversion images')
    aligned_images = []
    inversions_directory_path = args.save_dir
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(args.n_sample):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)
        aligned_images.append((imgs[0]+1)/2*255.)
    return aligned_images

def run_alignment(image_path, parse_net):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor, parse_net=parse_net)
    print("Aligned image: " + image_path)
    return aligned_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", type=str, default='cuda', help="Inference on CPU or GPU")
    parser.add_argument("--images_dir", type=str, default='input',
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default='output',
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--no_align", action="store_true", help="align face images before inference")
    parser.add_argument("--checkpoint_path_E", default="pretrained_models/encoder.pt", 
                            help="path to encoder checkpoint, optional: [encoder|encoder_without_pos].pt" )
    parser.add_argument("--checkpoint_path_D", default="pretrained_models/projector_WestEuropean.pt", 
                            help="path to decoder checkpoint, optional: projector_[WestEuropean|EastAsian|NorthAfrican].pt")
    parser.add_argument("--checkpoint_path_P", default="pretrained_models/79999_iter.pth", help="path to parser checkpoint")

    args = parser.parse_args()
    main(args)
