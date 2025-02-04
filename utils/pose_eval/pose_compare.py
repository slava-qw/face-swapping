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

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import hopenet as hopenet1
import torchvision
from torchvision import models
import torchvision.transforms as TF

from iutils import get_pose_emb

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from natsort import natsorted

# from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=20,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--path', type=str, nargs=2,
                    default=['data_imgs/CelebAMask-HQ/CelebA-HQ-img', 'data_imgs/generated_imgs'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

        self.transform_hopenet = torchvision.transforms.Compose([TF.ToTensor(), TF.Resize(size=(224, 224)),
                                                                 TF.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        image = self.transform_hopenet(Image.open(path).convert('RGB'))
        return image


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 99

    return degree


def compute_features(files, model, output_file, batch_size=50, device='cpu', num_workers=1):
    """
    Calculates the activations of the model and writes them to a file.

    Params:
    -- files       : List of image file paths
    -- model       : Instance of the model
    -- output_file : Path to the file where results will be stored (CSV format)
    -- batch_size  : Batch size for processing
    -- dims        : Dimensionality of model output
    -- device      : Device to run inference on
    -- num_workers : Number of parallel DataLoader workers
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

    with open(output_file, 'w') as f:
        f.write("yaw,pitch,roll\n")  # Writing the header

    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            yaw_gt, pitch_gt, roll_gt = model(batch)
            yaw_gt = headpose_pred_to_degree(yaw_gt).cpu().numpy().reshape(-1, 1)
            pitch_gt = headpose_pred_to_degree(pitch_gt).cpu().numpy().reshape(-1, 1)
            roll_gt = headpose_pred_to_degree(roll_gt).cpu().numpy().reshape(-1, 1)

        results = np.concatenate((yaw_gt, pitch_gt, roll_gt), axis=1)

        df = pd.DataFrame(results, columns=["yaw", "pitch", "roll"])
        df.to_csv(output_file, mode='a', header=False, index=False)  # Append to the file

    return pd.read_csv(output_file).to_numpy()


def compute_features_wrapp(path, model, batch_size, device,
                           num_workers=1):
    path = Path(path)
    files = natsorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
    tgt_ids = sorted([int(img_name.split('__tgt_')[1].split('.')[0]) for img_name in os.listdir(path)])
    output_file = "D:\PycharmProjects\gan_impl\megafs\data_imgs\gen_annotations.csv"
    pred_arr = compute_features(files, model, output_file, batch_size, device, num_workers)

    return pred_arr, tgt_ids


def calculate_id_given_paths(paths, batch_size, device, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    hopenet = hopenet1.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading hopenet')

    hopenet_state_dict = torch.load('hopenet_robust_alpha1.pkl')
    hopenet.load_state_dict(hopenet_state_dict)

    if torch.cuda.is_available():
        hopenet = hopenet.cuda()
        hopenet.eval()

    # TODO: if gen imgs amount are bigger, chane the ids slices
    feat1 = get_pose_emb()
    feat2, gen_ids = compute_features_wrapp(paths[1], hopenet, batch_size, device, num_workers)

    assert feat1[gen_ids].shape == feat2.shape, rf"{feat1[gen_ids].shape=} \neq {feat2.shape =}"
    feat1 = feat1[gen_ids]
    # find l2 distance
    dist = np.linalg.norm(feat1 - feat2, axis=1)
    value = np.mean(dist)

    return value


def main():
    print('aaa')
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        # num_avail_cpus = len(os.sched_getaffinity(0))
        num_avail_cpus = os.cpu_count()
        num_workers = min(num_avail_cpus, 8)

    else:
        num_workers = args.num_workers

    lib_path = Path(os.getcwd()).parent.parent
    args.path = list(map(lambda x: lib_path / Path(x), args.path))

    pose_value = calculate_id_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          num_workers)
    print('Pose_value: ', pose_value)


if __name__ == '__main__':
    main()
