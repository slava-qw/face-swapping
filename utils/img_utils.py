from IPython.display import display
import ipywidgets as widgets
from PIL import Image


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
