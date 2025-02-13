In this part of repo we will be using MegaFS for face swapping. The inference of the original model can be found int the `.inference/` folder or in official repo.

Additionally, as a way to improve this method we will try to replace the  Face Transfer Module (FTM) in stage II with handwritten Multi Head Cross Attention, to observe how this type of blending face representations affects on general result.
Thus, firstly, need to train this fusion block, so we will be using `.train/` folder.