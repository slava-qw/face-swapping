import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from megafs import HieRFE, FaceTransferModule, Generator, resnet50
from dataset import FaceDataset  # Assume we create a dataset class for CelebA-HQ
from loss_models import get_feature_extractor, get_id_extractor, get_landmark_detector


def train_hierfe(model, dataloader, optimizer, device):
    model.train()
    feature_extractor = get_feature_extractor()
    id_extractor = get_id_extractor()
    landmark_detector = get_landmark_detector()

    for batch in dataloader:
        images = batch.to(device)
        optimizer.zero_grad()
        encoded, _ = model(images)
        decoded, _ = generator(_, [encoded, None], randomize_noise=False)

        l_rec = nn.MSELoss()(images, decoded)
        l_lpips = nn.MSELoss()(feature_extractor(images), feature_extractor(decoded))
        l_id = 1 - nn.CosineSimilarity()(id_extractor(images), id_extractor(decoded))
        l_ldm = nn.MSELoss()(landmark_detector(images), landmark_detector(decoded))

        loss = l_rec + 0.8 * l_lpips + l_id + 1000 * l_ldm
        loss.backward()
        optimizer.step()


def train_ftm(ftm, dataloader, optimizer, device):
    ftm.train()

    feature_extractor = get_feature_extractor()
    id_extractor = get_id_extractor()
    landmark_detector = get_landmark_detector()

    for batch in dataloader:
        src, tgt = batch["source"].to(device), batch["target"].to(device)
        optimizer.zero_grad()
        src_enc, _ = hierfe(src)
        tgt_enc, _ = hierfe(tgt)
        swapped_enc = ftm(src_enc, tgt_enc)
        swapped_face, _ = generator(_, [swapped_enc, None], randomize_noise=False)

        l_rec = nn.MSELoss()(src, swapped_face)
        l_lpips = nn.MSELoss()(feature_extractor(tgt), feature_extractor(swapped_face))
        l_id = 1 - nn.CosineSimilarity()(id_extractor(src), id_extractor(swapped_face))
        l_ldm = nn.MSELoss()(landmark_detector(tgt), landmark_detector(swapped_face))
        l_norm = nn.MSELoss()(src_enc, swapped_enc)

        loss = 8 * l_rec + 32 * l_lpips + 24 * l_id + 100000 * l_ldm + 32 * l_norm
        loss.backward()
        optimizer.step()


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = FaceDataset("/path/to/dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Initialize models
hierfe = HieRFE(resnet50(False), num_latents=[4, 6, 8], depth=50).to(device)
ftm = FaceTransferModule(num_blocks=3, swap_indice=4, num_latents=18, typ="ftm").to(device)
generator = Generator(1024, 512, 8, channel_multiplier=2).to(device)

# Define optimizers
optimizer_hierfe = optim.Adam(hierfe.parameters(), lr=0.01)
optimizer_ftm = optim.Adam(ftm.parameters(), lr=0.01)

# Train HieRFE
train_hierfe(hierfe, dataloader, optimizer_hierfe, device)

# Train FTM
train_ftm(ftm, dataloader, optimizer_ftm, device)
