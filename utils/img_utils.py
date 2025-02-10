import gc
import os
import pathlib
import random

import cv2
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from IPython.display import display
import ipywidgets as widgets
from PIL import Image
from tqdm import tqdm


def show_faces(path):
    img = Image.open(path)

    src_id = path.split('/')[-1].split('_src_')[-1].split('__')[0]
    tgt_id = path.split('/')[-1].split('__tgt_')[-1].split('.')[0]

    make_grid = widgets.Output()
    with make_grid:
        display(img.resize((224, 224)))
        display(Image.open(f"./CelebAMask-HQ/CelebA-HQ-img/{src_id}.jpg").resize((224, 224)))
        display(Image.open(f"./CelebAMask-HQ/CelebA-HQ-img/{tgt_id}.jpg").resize((224, 224)))
    display(make_grid)


class MyImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, pairs, transforms=transforms.Compose([transforms.Resize((256, 256)),
                                                                       transforms.CenterCrop((256, 256)),
                                                                       transforms.ToTensor()])):
        self.img_root = img_root
        self.pairs = pairs
        self.transform = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src_id, tgt_id = self.pairs[index]
        s_path = os.path.join(self.img_root, f"{src_id}.jpg")
        t_path = os.path.join(self.img_root, f"{tgt_id}.jpg")

        # s_img = Image.open(s_path)
        # t_img = Image.open(t_path)

        s_img = cv2.cvtColor(cv2.imread(s_path), cv2.COLOR_BGR2RGB)
        t_img = cv2.cvtColor(cv2.imread(t_path), cv2.COLOR_BGR2RGB)

        # s_img = s_img.convert('RGB')
        # t_img = t_img.convert('RGB')

        if self.transform is not None:
            s_img = self.transform(s_img)
            t_img = self.transform(t_img)

        return {'tgt_img': t_img,
                'src_img': s_img,
                "src_id": src_id, "tgt_id": tgt_id,
                "src_path": s_path, "tgt_path": t_path}


model_name = "GAN"
input_dir = "./CelebAMask-HQ/CelebA-HQ-img"  # Directory with source-target pairs
output_dir = "./{model_name}_gen_10k/"  # Where to save results

shape_rate = 1.0  # 3D similarity [0.0-1.0]
id_rate = 1.0  # ID similarity [0.0-1.0]
iterations = 1  # [1-10]


def process_batch():
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    N = int(10000 * 3)
    rd1 = random.sample(range(0, 30000), N)
    rd2 = random.sample(range(0, 30000), N)

    pairs = [(str(i), str(j)) for i, j in zip(rd1, rd2)]

    dataset = MyImagePathDataset(img_root=input_dir,
                                 pairs=pairs,
                                 transforms=None)

    dataloader = DataLoader(dataset, batch_size=1)

    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        if count >= 10000:
            break

        src_img = batch['src_img'][0].numpy()
        tgt_img = batch['tgt_img'][0].numpy()
        tgt_id = batch['tgt_id']
        src_id = batch['src_id']
        # print(batch['tgt_id'], batch['src_id'])

        output_path = os.path.join(output_dir, f"{model_name}_src_{src_id[0]}__tgt_{tgt_id[0]}.jpg")

        # obtain and save result:
        # ...

        gc.collect()
        torch.cuda.empty_cache()


def parse_mask_path(path: str, mask_type: str = "src") -> str:
    idx_s = path.split('/')[-1].split('.')[0].split('_')
    idx = idx_s[2 if mask_type == 'src' else 5]
    idx = "0" * (5 - len(idx)) + idx
    mask_path = os.path.join("./content/masks_output", f"{idx}_mask.png")
    return mask_path
