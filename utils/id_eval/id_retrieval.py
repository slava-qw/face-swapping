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
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
from PIL import Image
# from model_irse import Backbone
import torchvision

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
import cv2
import albumentations as A
import torch.nn as nn
from natsort import natsorted
import pandas as pd

# from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dataset', type=str, default='celeba', help='Dataset to use')
parser.add_argument('--mask', type=bool, default=True,
                    help='whether to use mask or not')
parser.add_argument('--path', type=str, nargs=4,
                    default=['/content/CelebAMask-HQ/CelebA-HQ-img', '/content/gen_simswap_10k',
                             '/content/content/masks_output', '/content/content/masks_output'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

parser.add_argument('--print_sim', type=bool, default=False, )
parser.add_argument('--arcface', type=bool, default=True)
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def un_norm_clip(x1):
    x = x1 * 1.0  # to avoid changing the original tensor or clone() can be used
    reduce = False
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        reduce = True
    x[:, 0, :, :] = x[:, 0, :, :] * 0.26862954 + 0.48145466
    x[:, 1, :, :] = x[:, 1, :, :] * 0.26130258 + 0.4578275
    x[:, 2, :, :] = x[:, 2, :, :] * 0.27577711 + 0.40821073

    if reduce:
        x = x.squeeze(0)
    return x


def parse_mask_path(path: str, mask_type: str = "src") -> str:
    idx_s = path.split('/')[-1].split('.')[0].split('_')
    idx = idx_s[2 if mask_type == 'src' else 5]
    idx = "0" * (5 - len(idx)) + idx
    mask_path = os.path.join("/content/masks_output", f"{idx}_mask.png")
    return mask_path


class IDLoss(nn.Module):
    def __init__(self, multiscale=True):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        # self.opts = opts
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')

        self.facenet.load_state_dict(
            torch.load("/content/ir_se50.pth", map_location='cpu'))

        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        self.set_requires_grad(False)

    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag

    def extract_feats(self, x, clip_img=False):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TFF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x) if x.shape[2] != 256 else x  # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x)  # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale)

        return x_feats

    def forward(self, x, clip_img=False):
        x_feats_ms = self.extract_feats(x, clip_img=clip_img)
        return x_feats_ms[-1]


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


class MaskedImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, mask_type, maskfiles=None, transforms=None):
        self.files = files
        self.mask_type = mask_type
        self.maskfiles = maskfiles

        self.transforms = transforms
        self.trans = A.Compose([
            A.Resize(height=112, width=112)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_type == 'src':
            mask_path = self.maskfiles[i]
        else:
            mask_path = parse_mask_path(str(path), self.mask_type)

        ref_mask_img = Image.open(mask_path).convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask_img] = 255

        ref_converted_mask = Image.fromarray(ref_converted_mask).convert('L')
        reference_mask_tensor = get_tensor(normalize=False, toTensor=True)(ref_converted_mask)

        mask_ref = TF.Resize((112, 112))(reference_mask_tensor)

        ref_img = self.trans(image=image)
        ref_img = Image.fromarray(ref_img["image"])
        ref_img = get_tensor()(ref_img)

        ref_img = ref_img * mask_ref
        image = ref_img.unsqueeze(0)

        return image


def compute_features(files, mask_files, mask_type, model, output_file, batch_size=50, device='cpu',
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
       :param mask_type:
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = MaskedImagePathDataset(files, mask_type, maskfiles=mask_files, transforms=TF.ToTensor())

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    for batch in tqdm(dataloader):
        batch = batch.to(device).squeeze(1)

        with torch.no_grad():
            pred = model(batch)

        pred = pred.cpu().numpy()

        df = pd.DataFrame(pred, columns=range(512))
        df.to_csv(output_file, mode='a', header=False, index=False)  # Append to the file

    return pd.read_csv(output_file).to_numpy()


def compute_features_wrapp(path, mask_path, IDLoss_model, batch_size, device, output_name, num_workers=1, ret_tgt=True):
    path = pathlib.Path(path)
    files = natsorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
    # breakpoint()
    mask_path = pathlib.Path(mask_path)
    mask_files = natsorted([file for ext in IMAGE_EXTENSIONS
                            for file in mask_path.glob('*.{}'.format(ext))])

    mask_type = 'src' if output_name.startswith("exp_real") else "tgt"
    # pred_arr = compute_features(files, mask_files, mask_type, IDLoss_model, output_name, batch_size, device,
                                # num_workers)
    pred_arr = pd.read_csv("/content/exp_real_simswap_ids.csv", header=None).to_numpy()
    if ret_tgt:
        pred_arr = compute_features(files, mask_files, mask_type, IDLoss_model, output_name, batch_size, device,
                                num_workers)
        # pred_arr = pd.read_csv("/content/exp_gen_simswap_ids.csv", header=None).to_numpy()
        ids = sorted([
            int(parse_mask_path(img_name, mask_type).split('.')[0].split("_")[1].split('/')[-1])
            for img_name in os.listdir(path)
            ])
        return pred_arr, ids

    return pred_arr


def calculate_id_given_paths(paths, batch_size, device, num_workers=1, args=None):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    print(args.device)
    if args.arcface:
        IDLoss_model = IDLoss().to(args.device)

    feat1 = compute_features_wrapp(paths[0], paths[2], IDLoss_model, batch_size, device,
                                   "exp_real__ids.csv", num_workers, ret_tgt=False)

    feat2, swap_lab = compute_features_wrapp(paths[1], paths[3], IDLoss_model, batch_size, device,
                                             "exp_gen_simswap_ids.csv", num_workers)

    # dot produc to get similarity
    dot_prod = np.dot(feat2, feat1.T)
    pred = np.argmax(dot_prod, axis=1)
    # find accuracy of top 1 and top 5
    top1 = np.sum(np.argmax(dot_prod, axis=1) == swap_lab) / len(swap_lab)

    print(top1)
    print(dot_prod.shape)
    # top5_predictions = np.argsort(dot_prod, axis=1)[:, -5:]  # Get indices of top-5 predictions
    # top5_correct = np.sum(np.any(top5_predictions == np.array(swap_lab)[:, np.newaxis], axis=1))
    # top5 = top5_correct / len(swap_lab)  # Top-5 accuracy
    top5 = None

    print()
    feat_sel = feat1[swap_lab]
    feat_sel = feat_sel / np.linalg.norm(feat_sel, axis=1, keepdims=True)
    feat2 = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True)
    similarities = np.diagonal(np.dot(feat_sel, feat2.T))

    mean_dot_prod = np.mean(similarities)

    return top1, top5, mean_dot_prod, similarities


def main():
    args = parser.parse_args([])
    args.print_sim = True

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = os.cpu_count()
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    top1, top5, mean_dot_prod, similarities = calculate_id_given_paths(args.path, args.batch_size, args.device,
                                                                       num_workers,
                                                                       args=args)

    print('Top-1 accuracy: {:.2f}%'.format(top1 * 100))
    print('Top-5 accuracy: {:.2f}%'.format(top5 * 100))
    print('Mean ID feat:  {:.2f}'.format(mean_dot_prod))
    print('Mean similarity:  {:.2f}'.format(np.mean(similarities)))

    if args.print_sim:
        print('Similarities: \n ')
        for i in range(len(similarities)):
            print(i, ":", similarities[i])


if __name__ == '__main__':
    main()
