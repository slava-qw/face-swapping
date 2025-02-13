import gc
import random

import cv2
import torch
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import torchvision.transforms.functional as tF


def show_faces(path):
    from IPython.display import display
    import ipywidgets as widgets

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
    def __init__(self, img_root, pairs, return_dict=False, transforms=None):
        self.img_root = img_root
        self.pairs = pairs
        self.transform = transforms
        self.return_dict = return_dict

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def preprocess(src, tgt):
        src = cv2.resize(src.copy(), (256, 256))
        tgt = cv2.resize(tgt.copy(), (256, 256))
        src = torch.from_numpy(src.transpose((2, 0, 1))).float().mul_(1 / 255.0)
        tgt = torch.from_numpy(tgt.transpose((2, 0, 1))).float().mul_(1 / 255.0)

        src = tF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        tgt = tF.normalize(tgt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)

        return src, tgt

    def __getitem__(self, index):
        src_id, tgt_id = self.pairs[index]
        s_path = os.path.join(self.img_root, f"{src_id}.jpg")
        t_path = os.path.join(self.img_root, f"{tgt_id}.jpg")

        s_img = cv2.cvtColor(cv2.imread(s_path), cv2.COLOR_BGR2RGB)
        t_img = cv2.cvtColor(cv2.imread(t_path), cv2.COLOR_BGR2RGB)

        s_img, t_img = self.preprocess(s_img, t_img)
        # src_face = cv2.imread(s_path)
        # tgt_face = cv2.imread(t_path)

        # s_img = src_face[:, :, ::-1]
        # t_img = tgt_face[:, :, ::-1]

        if self.transform is not None:
            s_img = self.transform(s_img)
            t_img = self.transform(t_img)

        if self.return_dict:
            return {'tgt_img': t_img,
                    'src_img': s_img,
                    "src_id": src_id, "tgt_id": tgt_id,
                    "src_path": s_path, "tgt_path": t_path}

        return s_img, t_img


def process_batch():
    model_name = "GAN"
    input_dir = "./CelebAMask-HQ/CelebA-HQ-img"  # Directory with source-target pairs
    output_dir = f"./{model_name}_gen_10k/"  # Where to save results

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


def make_grid():
    # Assuming you have the paths to your images
    main_images_path = "/content/SimSwap/gen_simswap_10k_224"
    secondary_images_path = '/content/SimSwap/gen_simswap_10k'
    output_path = '/content'

    main_data = os.listdir(main_images_path)
    main_images = [Image.open(os.path.join(main_images_path, name)) for name in main_data]

    secondary_images = {}
    for img_name in os.listdir(secondary_images_path):
        if img_name.endswith('.png'):
            secondary_images[img_name] = Image.open(os.path.join(secondary_images_path, img_name))

    grid_size = (6, 6)
    image_size = main_images[0].size
    grid_image = Image.new('RGB', (image_size[0] * grid_size[0], image_size[1] * grid_size[1]), color='white')

    draw = ImageDraw.Draw(grid_image)

    font = ImageFont.load_default(size=32)
    draw.text((10, image_size[0] // 2 + 5), "Source", fill="black", font=font)
    draw.text((image_size[0] // 2 + 5, 10), "Target", fill="black", font=font)
    draw.line([(0, 0), (image_size[0], image_size[1])], fill="black", width=2)

    for i, img in enumerate(main_images):
        grid_image.paste(img, (0, (i + 1) * image_size[1]))
        draw.rectangle([(0, (i + 1) * image_size[1]),
                        (image_size[0], (i + 2) * image_size[1])], outline="black", width=2)

        grid_image.paste(img, ((i + 1) * image_size[0], 0))
        draw.rectangle([((i + 1) * image_size[0], 0),
                        ((i + 2) * image_size[0], image_size[1])], outline="black", width=2)

    for row, src_id in zip(range(1, 6), main_data):
        for col, tgt_id in zip(range(1, 6), main_data):
            img_name = f'simswap_src_{src_id.split(".")[0]}__tgt_{tgt_id.split(".")[0]}.png'
            if img_name in secondary_images:
                grid_image.paste(secondary_images[img_name], (col * image_size[0], row * image_size[1]))
                draw.rectangle([(col * image_size[0], row * image_size[1]),
                                ((col + 1) * image_size[0], (row + 1) * image_size[1])], outline="black", width=2)

    grid_image.save(os.path.join(output_path, 'simswap_grid_bordered.pdf'))

    print("Grid image with borders saved successfully!")


def save_main_pairs():
    img_root = "/content/HiFiFace-pytorch/CelebAMask-HQ/CelebA-HQ-img"
    abs = [44, 1337, 24, 7331, 73]

    from PIL import Image
    import os

    os.makedirs("/content/results_main", exist_ok=True)

    for i in abs:
        img = Image.open(f"{img_root}/{i}.jpg")
        img.save(f"/content/results_main/{i}.png")
