import os
import PIL
import torch
from einops import rearrange
from tqdm import tqdm
from insightface.app import FaceAnalysis
from mixer import Mixer
import sys
import gc

sys.path.append('../stylegan2-ada-pytorch')

import dnnlib
import legacy
import numpy as np


def obtain_ws(G, batch_size=4, device='cuda'):
    lt = torch.randn([batch_size, G.z_dim], device=device)
    ls = torch.randn([batch_size, G.w_dim], device=device)

    lt = G.mapping(lt, None)  # already have made broadcasting inside
    ls = G.mapping(ls, None)  # already have made broadcasting inside

    return ls, lt


def gen_img_from_lat(G, ws, noise_mode='const', name='', outdir='./output', save=True):
    assert ws.shape[1:] == (G.num_ws, G.w_dim), "Shape mismatch for ws."

    imgs = G.synthesis(ws, noise_mode=noise_mode)

    if save:
        os.makedirs(outdir, exist_ok=True)
        for idx, img in enumerate(imgs):
            img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(f'{outdir}/{name}_proj{idx:02d}.png')

    return imgs


def prepare_models(device='cuda'):
    network_pkl = "stylegan2-ffhq-config-f.pkl"
    print(f'Loading networks from "{network_pkl}"...')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    mixer = Mixer().to(device)

    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(256, 256))
    # app = None
    return mixer, G, app


def get_emb(app, imgs):
    embds = []

    for img in imgs:
        img = rearrange(img, 'c h w -> h w c')
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        faces = app.get(img.cpu().numpy())
        if not faces:
            raise ValueError("No faces detected in the image.")

        largest_face = max(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))

        id_emb = torch.tensor(largest_face['embedding'], dtype=torch.float32, device=img.device).unsqueeze(0)
        id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)

        embds.append(id_emb)

    return torch.cat(embds, dim=0)


def train(num_steps=100, device='cuda'):
    mixer, G, app = prepare_models(device)

    optimizer = torch.optim.Adam(mixer.parameters(), lr=1e-4)

    for step in tqdm(range(num_steps), desc="Training Mixer"):
        ls, lt = obtain_ws(G, batch_size=2, device=device)
        l_sw = mixer(ls, lt)
        print("3", l_sw.shape)

        imgs_sw = gen_img_from_lat(G, l_sw, name='sw', save=True)
        imgs_s = gen_img_from_lat(G, ls, name='s', save=True)
        # for check
        imgs_t = gen_img_from_lat(G, lt, name='t', save=True)

        emb_s = get_emb(app, imgs_s)
        emb_sw = get_emb(app, imgs_sw)

        loss_id = (1 - torch.einsum("kj,kj->k", emb_s, emb_sw)).mean()
        loss_lp = torch.mean((lt - l_sw) ** 2) / lt.shape[-1]

        loss = loss_id + 1e2 * loss_lp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    # save
    torch.save(mixer.state_dict(), 'mixer.pth')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(num_steps=100, device=device)
