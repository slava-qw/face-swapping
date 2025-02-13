import os

import cv2
import torch
import torchvision.transforms.functional as tF


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

