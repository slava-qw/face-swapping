import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

from inference import MegaFS

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

    @staticmethod
    def find_path(file_end, dir_path="/content/CelebAMask-HQ/CelebAMask-HQ-mask-anno"):
        file_end += '.png'

        for root, dirs, files in os.walk(dir_path):
            for dir in dirs:
                for root, dirs, files in os.walk(os.path.join(dir_path, dir)):
                    for file in files:
                        if file.endswith(file_end):
                            return os.path.join(root, file)
        return None

    @staticmethod
    def load_array(path_to_file):
        im_frame = Image.open(path_to_file).convert('RGB')
        np_frame = np.array(im_frame.getdata())  # (H * W, C)
        return np_frame

    def encode_segmentation_rgb(self, tgt_idx, no_neck=True):
        face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]

        mouth_map = self.load_array(self.find_path(file_end=f"{tgt_idx}_mouth"))
        hair_map = self.load_array(self.find_path(file_end=f"{tgt_idx}_hair"))

        face_map = np.zeros_like(mouth_map)
        for valid_id in face_part_ids:
            attr_path = self.find_path(file_end=f"{tgt_idx}_{str2num[valid_id]}")
            if attr_path is None:
                continue
            face_map += self.load_array(attr_path)

        face_map = np.clip(face_map, 0, 255).astype(np.uint8)
        mouth_map = np.clip(mouth_map, 0, 255).astype(np.uint8)
        hair_map = np.clip(hair_map, 0, 255).astype(np.uint8)

        return np.stack([face_map, mouth_map, hair_map], axis=2)

    def read_pair(self, src_idx, tgt_idx):
        src_face = cv2.imread(os.path.join(self.img_root, "{}.jpg".format(src_idx)))
        tgt_face = cv2.imread(os.path.join(self.img_root, "{}.jpg".format(tgt_idx)))
        # tgt_mask  = cv2.imread(os.path.join(self.mask_root, "{}.png".format(tgt_idx)))

        src_face_rgb = src_face[:, :, ::-1]
        tgt_face_rgb = tgt_face[:, :, ::-1]
        tgt_mask = self.encode_segmentation_rgb(tgt_idx)
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
