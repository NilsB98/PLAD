import torch
import torch.nn as nn

import plotting.plotting
from model.plad import Classifier
from data.function_ds import LabeledDataset
from torch.utils.data import DataLoader
from utils.devices import get_device

loss_fn = nn.BCELoss()

device = get_device()
model = Classifier(2, device)
optimizer = torch.optim.Adam(model.parameters())

PATH = r'../checkpoints/classifier1.pt'


def train():
    batch_size = 2 ** 7
    dataset = LabeledDataset(2 ** 20)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        batch_loss = 0

        for i, (x_, y_) in enumerate(train_loader):
            x_, y_ = x_.to(device), y_.to(device)

            optimizer.zero_grad()
            outputs = model(x_)
            loss = loss_fn(outputs, y_.view(-1, 1).float())
            loss.backward()

            optimizer.step()
            batch_loss += loss.item()

        print(f"EPOCH {epoch + 1}: {batch_loss / batch_size}")

        with torch.no_grad():
            data, labels = next(iter(train_loader))
            normal_data = data[torch.logical_not(labels.bool())]
            perturbed_data = data[labels.bool()]
            plotting.plotting.plot_decision(f"Only classifier trained - Epoch {epoch + 1}", model, normal_data,
                                            perturbed_data)

        torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    train()
