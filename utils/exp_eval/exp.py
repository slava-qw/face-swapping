"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from pathlib import Path
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
from natsort import natsorted

from Deep3DFaceRecon_pytorch_edit.options.test_options import TestOptions
from Deep3DFaceRecon_pytorch_edit.models import create_model

# give empty string to use the default options
test_opt = TestOptions('')
test_opt = test_opt.parse()

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--path', type=str, nargs=2,
                    # default=['dataset/FaceData/CelebAMask-HQ/Val_target',
                    #          'results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16/results'],
                    default=['data_imgs/CelebAMask-HQ/CelebA-HQ-img', 'data_imgs/generated_imgs'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--print_sim', type=bool, default=False, )

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [TF.ToTensor()]

    if normalize:
        transform_list += [TF.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))]
    return TF.Compose(transform_list)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms
        self.transform_hopenet = TF.Compose([TF.ToTensor(), TF.Resize(size=(224, 224)),
                                             TF.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

        self.transform = TF.Compose(
            [TF.ToTensor(),
             TF.Resize(size=(224, 224)),
             TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        im = Image.open(path).convert('RGB')

        im = im.resize((512, 512), Image.BICUBIC)
        im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        return im


def compute_features(files, model, output_file, batch_size=50, device='cpu',
                     num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    for batch in tqdm(dataloader):
        batch = batch.to(device).squeeze(1)

        with torch.no_grad():
            coeff = model.forward(batch)
            pred = coeff['exp']

        pred = pred.cpu().numpy()

        df = pd.DataFrame(pred, columns=range(64))
        df.to_csv(output_file, mode='a', header=False, index=False)  # Append to the file

    return pd.read_csv(output_file).to_numpy()


def compute_features_wrapp(path, model, output_file, batch_size, device,
                           num_workers=1, ret_tgt=True):
    path = Path(path)
    files = natsorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])

    pred_arr = compute_features(files, model, output_file, batch_size, device, num_workers)

    if ret_tgt:
        tgt_ids = sorted([int(img_name.split('__tgt_')[1].split('.jpg')[0]) for img_name in os.listdir(path)])
        return pred_arr, tgt_ids

    return pred_arr


def calculate_id_given_paths(paths, batch_size, device, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    bfm_folder = 'exp_eval/Deep3DFaceRecon_pytorch_edit/BFM'
    abs_path = Path(os.getcwd()).parent
    path = abs_path / Path(bfm_folder)
    test_opt.bfm_folder = path

    test_opt.epoch = 20
    models_expression = create_model(test_opt)
    models_expression.setup(test_opt)

    if torch.cuda.is_available():
        models_expression.net_recon.cuda()
        models_expression.facemodel.to("cuda")
    models_expression.eval()

    feat1 = compute_features_wrapp(str(paths[0]), models_expression, "exp_real.csv", batch_size,
                                   device, num_workers, ret_tgt=False)
    feat2, swap_lab = compute_features_wrapp(str(paths[1]), models_expression, "exp_gen.csv", batch_size,
                                             device, num_workers)

    # feat1 is bigger than feat2
    feat1 = feat1[swap_lab]

    diff_feat = np.power(feat1 - feat2, 2)
    diff_feat = np.sum(diff_feat, axis=-1)

    value = np.sqrt(diff_feat)
    similarities = value
    value = np.mean(value)

    diff_feat_sq = np.sum(np.square(feat1 - feat2), axis=1)

    # Compute the L2 distance (Euclidean distance) by taking the square root of the sum of squared differences
    L2_distance = np.sqrt(diff_feat_sq)

    return value, similarities


def main():
    args = parser.parse_args()
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = os.cpu_count()
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    lib_path = Path(os.getcwd()).parent.parent
    args.path = list(map(lambda x: lib_path / Path(x), args.path))

    expression_value, similarities = calculate_id_given_paths(args.path,
                                                              args.batch_size,
                                                              device,
                                                              num_workers)
    print('expression_value: ', expression_value)

    if args.print_sim:
        print('Similarities: \n ')
        for i in range(len(similarities)):
            print(i, ":", similarities[i])


if __name__ == '__main__':
    main()
