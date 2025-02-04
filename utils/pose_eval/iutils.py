import os
import pandas as pd
from pathlib import Path


def prepare_path(rel_path: str):
    pass


def get_pose_emb():
    data_path = Path("data_imgs/CelebAMask-HQ/CelebAMask-HQ-pose-anno.txt")
    current_dir = Path(os.getcwd()).parent.parent
    data_path = current_dir / data_path

    with open(data_path, 'r') as f:
        n = int(next(f))
        assert n == 30000, "not all data from CelebA-HQ are included"

        df = pd.read_csv(f, sep=' ')
        pose_emb = df.to_numpy()

    return pose_emb
