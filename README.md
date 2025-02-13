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

| Method    | FID-10k, | LPIPS | Pose  | Exp. | Mean ID sim. |
|-----------|----------|---|---|------|--------------|
| MegaFS    | 44.89    | 0.49  | 20.57 | 2.43 | 0.94         |
| SimSwap   | 28.92    | 0.31  | 20.95 | 2.45 | 0.94         |
| HifiFace* | 21.61    |   0.069    |    18.94   | 2.27 | 0.93         |

*are calculated with 1k sample of generated images from Celeba-HQ dataset due to time constraints and face detection issues.

The values calculated in the comparison table were extracted via pretrained models. All instructions for usage and weights can be found in the links below:
- FID: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- LPIPS: [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- Pose:  Bardia Doosti, et al. "Hope-net: A graph-based model for hand-object
pose estimation". In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 6608–
6617, 2020. ([url](https://www.researchgate.net/publication/340374324_HOPE-Net_A_Graph-based_Model_for_Hand-Object_Pose_Estimation))
- ID similarity:  [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) from Yu Deng, et al. "Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set".
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019.

For the pose and expression the L2 distance was employed to compare the vectorized representations of 
target image and swapped image.

## TODO:
- [ ] fix all paths
