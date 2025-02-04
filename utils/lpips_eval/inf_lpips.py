import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from lpips import LPIPS


class LPIPSDataset(Dataset):
    image_ext = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                 'tif', 'tiff', 'webp'}

    def __init__(self, folder_1, folder_2, transform=None):
        """
        Args:
            folder_1 (str): Path to the first dataset folder (e.g., /kaggle/working/results)
            folder_2 (str): Path to the second dataset folder (e.g., /kaggle/input/celebamask-hq/CelebAMask-HQ/CelebA-HQ-img)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_1 = folder_1
        self.folder_2 = folder_2
        self.transform = transform

        self.images_1 = [f for f in os.listdir(folder_1) if f.split('.')[-1] in self.image_ext]

    def __len__(self):
        return len(self.images_1)

    def __getitem__(self, idx):
        img_1_name = self.images_1[idx]
        tgt_idx = img_1_name.split('__tgt_')[1].split('.')[0]
        img_type = img_1_name.split('.')[-1]
        img_2_name = f"{int(tgt_idx)}.jpg"  # format of CelebA-HQ dataset is .jpg

        img_1_path = os.path.join(self.folder_1, img_1_name)
        img_2_path = os.path.join(self.folder_2, img_2_name)

        img_1 = Image.open(img_1_path).convert("RGB")
        img_2 = Image.open(img_2_path).convert("RGB")

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        assert img_1.shape == img_2.shape, rf"{img_1.shape =} \neq {img_2.shape =}"

        return img_1, img_2, tgt_idx


transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = LPIPSDataset(
    folder_1="/kaggle/working/results",
    folder_2="/kaggle/input/celebamask-hq/CelebAMask-HQ/CelebA-HQ-img",
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LPIPS().to(device)

total_lpips = 0.0
for batch_idx, (img_1, img_2, tgt_idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
    img_1 = img_1.cuda()
    img_2 = img_2.cuda()

    lpips_value = model(img_1, img_2)
    total_lpips += lpips_value.item()

    # Optionally print intermediate results
    # print(f"Batch {batch_idx}, LPIPS: {lpips_value.item()}")

average_lpips = total_lpips / len(dataloader)
print(f"Final LPIPS: {average_lpips}")
