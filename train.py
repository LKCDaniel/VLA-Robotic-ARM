import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VisionActionModel
from dataset import PickPlaceDataset
from tqdm import tqdm


def train():
    device = "cuda"
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3
    checkpoint_pth = "checkpoint.pth"
    stat_path = "stat.npz"
    save_every = 200

    model = VisionActionModel().to(device)
    model.train()
    if os.path.exists(checkpoint_pth):
        model_state = torch.load(checkpoint_pth)
        model.load_state_dict(model_state)

    dataset = PickPlaceDataset(json_path="train_data.json", stat_path=stat_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    count = 0
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc=f"Epoch {epoch + 1}/{num_epochs}",
                            ncols=100)

        total_loss = 0
        for i, (img1, img2, img3, current_state, next_state_gt) in progress_bar:
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            current_state = current_state.to(device)
            next_state_gt = next_state_gt.to(device)

            optimizer.zero_grad()
            next_state_pd = model(img1, img2, img3, current_state)
            loss = nn.L1Loss()(next_state_pd, next_state_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}"
            })

            if count > 0 and (count % save_every == 0):
                torch.save(model.state_dict(), checkpoint_pth)
            count += 1


if __name__ == '__main__':
    train()
