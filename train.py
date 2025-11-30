import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VisionActionModel
from dataset import PickPlaceDataset
from tqdm import tqdm


@torch.no_grad()
def validate(model, dataloader_validate, device):
    model.eval()
    loss_list = []
    for img1, img2, img3, current_state, action_gt in dataloader_validate:
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        current_state = current_state.to(device)
        action_gt = action_gt.to(device)
        action_pd = model(img1, img2, img3, current_state)
        loss = nn.L1Loss()(action_pd, action_gt)
        loss_list.append(loss.item())
    model.train()
    return sum(loss_list) / len(loss_list)


def train():
    device = "cuda"
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3
    checkpoint_pth = "checkpoint.pth"
    save_every = 600
    validate_every = 300

    model = VisionActionModel().to(device)
    model.train()
    if os.path.exists(checkpoint_pth):
        model_state = torch.load(checkpoint_pth)
        model.load_state_dict(model_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset_train = PickPlaceDataset(data_path="data_train.json")
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)

    dataset_validate = PickPlaceDataset(data_path="data_validate.json")
    dataloader_validate = DataLoader(dataset=dataset_validate, batch_size=batch_size, shuffle=True, drop_last=False)

    count = 0
    validation_loss = validate(model, dataloader_validate, device)
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader_train),
                            total=len(dataloader_train),
                            desc=f"Epoch {epoch + 1}/{num_epochs}",
                            ncols=120)

        total_loss = 0
        for i, (img1, img2, img3, current_state, action_gt) in progress_bar:
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            current_state = current_state.to(device)
            action_gt = action_gt.to(device)

            optimizer.zero_grad()
            action_pd = model(img1, img2, img3, current_state)
            loss = nn.L1Loss()(action_pd, action_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix({
                "iter_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "valid_loss": f"{validation_loss:.4f}"
            })

            if count > 0 and (count % save_every == 0):
                torch.save(model.state_dict(), checkpoint_pth)

            if count > 0 and (count % validate_every == 0):
                validation_loss=validate(model, dataloader_validate, device)

            count += 1


if __name__ == '__main__':
    train()
