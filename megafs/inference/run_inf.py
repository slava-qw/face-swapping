import os
import cv2
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tF
import torch
import gc
from inference import MegaFS


class FaceSwapDataset(Dataset):
    def __init__(self, img_root, mask_root, pairs):
        self.img_root = img_root
        self.mask_root = mask_root
        self.pairs = pairs  # List of (src_idx, tgt_idx)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_idx, tgt_idx = self.pairs[idx]

        src_face = cv2.imread(os.path.join(self.img_root, f"{src_idx}.jpg"))
        tgt_face = cv2.imread(os.path.join(self.img_root, f"{tgt_idx}.jpg"))

        src_face_rgb = cv2.cvtColor(src_face, cv2.COLOR_BGR2RGB)
        tgt_face_rgb = cv2.cvtColor(tgt_face, cv2.COLOR_BGR2RGB)

        tgt_idx = "0" * (5 - len(str(tgt_idx))) + str(tgt_idx)
        tgt_mask = cv2.imread(os.path.join(self.mask_root, f"{tgt_idx}_mask.png"))

        # https://discuss.pytorch.org/t/negative-strides-in-tensor-error/134287/2
        src_face_rgb = np.ascontiguousarray(src_face[:, :, ::-1])  # Copy ensures no negative strides
        tgt_face_rgb = np.ascontiguousarray(tgt_face[:, :, ::-1])
        tgt_mask = np.ascontiguousarray(tgt_mask)

        src_face_rgb_ = cv2.resize(src_face_rgb.copy(), (256, 256))
        tgt_face_rgb_ = cv2.resize(tgt_face_rgb.copy(), (256, 256))
        src = torch.from_numpy(src_face_rgb_.transpose((2, 0, 1))).float().mul_(1 / 255.0)
        tgt = torch.from_numpy(tgt_face_rgb_.transpose((2, 0, 1))).float().mul_(1 / 255.0)

        src = tF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        tgt = tF.normalize(tgt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)

        return src_idx, tgt_idx, src_face_rgb, tgt_face_rgb, tgt_mask, src, tgt


def process_batch(batch, model, save_path):
    src_idx, tgt_idx, src_face_rgb, tgt_face_rgb, tgt_mask, source, target = batch
    # source, target = model.preprocess(src_face_rgb, tgt_face_rgb)

    swapped_face = model.swap(source, target)
    swapped_face = model.postprocess(swapped_face, tgt_face_rgb.numpy(), ~tgt_mask.squeeze(0).numpy())

    # result = np.hstack((src_face_rgb.numpy()[:, :, ::-1], tgt_face_rgb.numpy()[:, :, ::-1], swapped_face))
    # print(f"{result.shape=}")
    print(swapped_face.shape)
    print(src_idx[0], tgt_idx[0])
    save_file = f"{save_path}/{model.swap_type}_src_{src_idx[0]}__tgt_{tgt_idx[0]}.jpg"
    cv2.imwrite(save_file, swapped_face[0][:, :, ::-1])


def run_in_batches(handler, img_root, mask_root, pairs, batch_size=1, num_workers=4, save_path="/content/results"):
    assert batch_size == 1, "Batch size must be 1"
    os.makedirs(save_path, exist_ok=True)

    dataset = FaceSwapDataset(img_root, mask_root, pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for batch in tqdm(dataloader):
        process_batch(batch, handler, save_path)

        gc.collect()
        torch.cuda.empty_cache()


str2num = {
    # 0: 'background',
    1: 'skin',
    2: 'l_brow',
    3: 'r_brow',
    4: 'l_eye',
    5: 'r_eye',
    6: 'eye_g',
    7: 'l_ear',
    8: 'r_ear',
    # 9: 'ear_r',
    10: 'nose',
    11: 'mouth',
    12: 'u_lip',
    13: 'l_lip',
    14: 'neck',
    # 15: 'neck_l',
    # 16: 'cloth',
    17: 'hair',
    # 18: 'hat'
}


class My_MegaFS(MegaFS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_pair(self, src_idx, tgt_idx):
        src_face = cv2.imread(os.path.join(self.img_root, "{}.jpg".format(src_idx)))
        tgt_face = cv2.imread(os.path.join(self.img_root, "{}.jpg".format(tgt_idx)))
        tgt_mask = cv2.imread(os.path.join(self.mask_root, "{}_mask.png".format(tgt_idx)))

        src_face_rgb = src_face[:, :, ::-1]
        tgt_face_rgb = tgt_face[:, :, ::-1]
        # tgt_mask = self.encode_segmentation_rgb(tgt_idx)

        return src_face_rgb, tgt_face_rgb, tgt_mask

    def run(self, src_idx, tgt_idx, refine=False, save_path='/content'):
        src_face_rgb, tgt_face_rgb, tgt_mask = self.read_pair(src_idx, tgt_idx)
        source, target = self.preprocess(src_face_rgb, tgt_face_rgb)
        swapped_face = self.swap(source, target)
        swapped_face = self.postprocess(swapped_face, tgt_face_rgb, tgt_mask)

        result = np.hstack((src_face_rgb[:, :, ::-1], tgt_face_rgb[:, :, ::-1], swapped_face))

        if refine:  # True give some bad results...
            swapped_tensor, _ = self.preprocess(swapped_face[:, :, ::-1], swapped_face)
            refined_face = self.refine(swapped_tensor)
            refined_face = self.postprocess(refined_face, tgt_face_rgb, tgt_mask)
            result = np.hstack((result, refined_face))

        save_path = f"{save_path}/{self.swap_type}_{src_idx}_and_{tgt_idx}"
        print(save_path)
        cv2.imwrite("{}.jpg".format(save_path), result)


if __name__ == "__main__":
    import random

    # TODO: fix all paths
    img_root = "/content/CelebAMask-HQ/CelebA-HQ-img"
    mask_root = "/content/content/masks_output"

    handler = My_MegaFS(swap_type="ftm",
                        img_root=img_root,
                        mask_root=mask_root)

    N = 10000
    rd1 = random.sample(range(0, 30000), N)
    rd2 = random.sample(range(0, 30000), N)

    pairs = [(str(i), str(j)) for i, j in zip(rd1, rd2)]
    run_in_batches(handler, img_root, mask_root, pairs, batch_size=1, num_workers=4)
