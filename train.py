import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VisionActionModel
from dataset import PickPlaceDataset
from tqdm import tqdm


def train():
    device = "cuda"
    batch_size = 128
    num_epochs = 200
    learning_rate = 1e-3
    save_every = 1
    checkpoint_pth = "checkpoint.pth"
    stat_path = "stat.npz"

    model = VisionActionModel().to(device)
    model.train()
    if os.path.exists(checkpoint_pth):
        model_state = torch.load(checkpoint_pth)
        model.load_state_dict(model_state)

    dataset = PickPlaceDataset(npz_path="data.npz", stat_path=stat_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    for epoch in range(num_epochs):
        loss_list = []
        loss_cache = 0

        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc=f"Epoch {epoch + 1}/{num_epochs}",
                            ncols=100)

        for i, (img1, img2, state, action_gt) in progress_bar:
            img1 = img1.to(device)
            img2 = img2.to(device)
            state = state.to(device)
            action_gt = action_gt.to(device).float()

            optimizer.zero_grad()

            action_pd = model(img1, img2, state)

            loss = criterion(action_pd, action_gt)

            loss.backward()
            optimizer.step()

            loss_cache += loss.item()
            loss_list.append(loss.item())

            # 实时更新进度条上的 loss
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                'avg_loss': f"{loss_cache / (i + 1):.4f}"
            })

        avg_loss = sum(loss_list) / len(loss_list)
        print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f}")

        if epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_pth)
            print("model weights saved at", checkpoint_pth)


if __name__ == '__main__':
    train()