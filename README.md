# Basic code for inference of GANs face swap methods.

In this repo I include some basic code for inference of GANs face swap methods:
- MegaFS ([arxiv](http://arxiv.org/abs/2105.04932), [github](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels))
- SimSwap ([arxiv](https://arxiv.org/pdf/2106.06340v1.pdf), [github](https://github.com/neuralchen/SimSwap))
- HifiFace ([arxiv](https://arxiv.org/pdf/2106.09965), [github](https://github.com/xuehy/HiFiFace-pytorch/tree/Main))

Some parts are borrowed from un/officials repos of these articles. 

Right now this repo includes only inference codes and quantitative evaluation of generated images via such metrics as FID-10k, LPIPS, and pose simmilarity (code partially borrowed from [[GiHub]](https://github.com/Sanoojan/REFace) and modified for particular dataset).
Shortly, I will add my own improvements of these baseline and compare them in terms of aforementioned metrics.

All metrics are evaluated on Celeba-HQ (1024x1024) dataset, sample of 30k images of CelebA dataset ([link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).
The amount of generated images used in calculations is 10k.

All additional information (like, weights of pretrained models, and folder structure) to run the inference code can be found on official pages (see corresponding links) and in the inference notebooks.

Although many methods use their own models architectures (trained from scratch), recent works follow the paradigm of using pre-trained large models for high-resolution conditional image generation, while training only small id-retrieval / swapping networks.
And the most popular of them is StyleGAN2 ([sourece code](https://github.com/NVlabs/stylegan2-ada-pytorch)).

Own metrics evaluation:

| Method    | FID-10k | LPIPS | Pose  | Exp. | ID retr. / sim |
|-----------|---------|---|---|---|---|
| MegaFS    | 44.89   | 0.49  | 20.57 |      | |
| SimSwap   | 28.92   | 0.31  | 20.95 |      |  |
| HifiFace* | 21.61   |   0.069    |    18.94   |      |  |
| GHOST     |         |       |    |      |  |

*are calculated with 1k sample of generated images fromCeleba-HQ dataset due to time constraints and face detection issues.

## TODO:
- [ ] fix all paths
- [ ] fix id retr./simm. and expression metrics for evaluation
- [ ] add comparison table or link to results
