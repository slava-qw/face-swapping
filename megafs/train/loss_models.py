import torch
from losses.arcface import iresnet100
import torch.nn as nn
import torch.nn.functional as F

from losses.detect import detect_landmarks
from losses.FAN import FAN


class IdentityLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = iresnet100(fp16=False)
        self.model.load_state_dict(torch.load('losses/pretrained_ckpts/arcface.pt', map_location=device))
        self.model.to(device).eval()

    def forward(self, source, swap):
        source = self.model(F.interpolate(source, [112, 112], mode='bilinear', align_corners=False))
        swap = self.model(F.interpolate(swap, [112, 112], mode='bilinear', align_corners=False))
        return 1 - nn.CosineSimilarity()(source, swap, dim=1)


class LandmarkLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = FAN(4, "False", "False", 98)
        self.setup_model('losses/pretrained_ckpts/WFLW_4HG.pth')
        self._mse = torch.nn.MSELoss(reduction='none')

    def setup_model(self, path_to_model: str):
        checkpoint = torch.load(path_to_model, map_location='cpu')
        if 'state_dict' not in checkpoint:
            self.model.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = self.model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            self.model.load_state_dict(model_weights)
        self.model.eval().to(self.device)

    def forward(self, target, swap):
        swap = torch.nn.functional.interpolate(swap, [256, 256], mode='bilinear', align_corners=False)
        target = torch.nn.functional.interpolate(target, [256, 256], mode='bilinear', align_corners=False)

        swap_lmk, _ = detect_landmarks(swap, self.model, normalize=True)
        target_lmk, _ = detect_landmarks(target, self.model, normalize=True)

        return nn.MSELoss()(swap_lmk - target_lmk)