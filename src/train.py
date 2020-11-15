from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os

from model import SusNetv2

def train():
    os.environ["L5KIT_DATA_FOLDER"] = "../"
    dm = LocalDataManager(None)
    cfg = load_config_data("../agent_motion_config.yaml")

    train_cfg = cfg['train_data_loader']
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg['key'])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset,
                        shuffle=train_cfg['shuffle'],
                        batch_size=train_cfg['batch_size'],
                        num_workers=train_cfg['num_workers'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SusNetv2(cfg['history_num_frames'],cfg['future_num_frames'])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')
    train_it = iter(train_dataloader)

    losses =[]
    progress_bar = tqdm(range(cfg['train_params']['max_num_steps']))
    for step in progress_bar:
        try:
            data=next(train_it)
        except StopIteration:
            train_it = iter(train_dataloader)
            data = next(train_it)

        model.train()
        torch.set_grad_enabled(True)

        inputs = data['image'].to(device)
        target_availabilities = data['target_availabilities'].unsqueeze(-1).to(device)
        targets = data['target_positions'].to(device)
        outputs = model(inputs).reshape(targets.shape)
        loss = criterion(outputs, targets)
        loss = loss * target_availabilities
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        progress_bar.set_description(f'Loss: {loss.item()}, loss(avg): {np.mean(losses_train)}')
    plt.plot(np.arange(len(losses)),losses, label='Train loss')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(),'lyft_model.pth')

if __name__=='__main__':
    train()
