import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import plotting.plotting
from data.function_ds import NormalDataset
from model.plad import PLAD, Classifier
from utils.devices import get_device


def L(pred_normal, pred_pert, alpha, beta) -> Tensor:
    """
    Loss used to optimize PLAD with:
    y_true = 0
    y_pert = 1

    :param pred_normal: predictions for the normal data points
    :param pred_pert: predictions for the perturbed data points
    :param alpha: alpha calculated by the generator
    :param beta: beta calculated by the generator
    :return:
    """

    lambd = 0.01
    loss_fn = nn.BCELoss()

    l = loss_fn(pred_pert, torch.ones_like(pred_pert)) + lambd * (torch.norm(alpha - 1) + torch.norm(beta))

    if not use_pretrained_classifier:
        l += loss_fn(pred_normal, torch.zeros_like(pred_normal))

    return l


PATH_PRETRAINED_CLS = r'../checkpoints/classifier.pt'
use_pretrained_classifier = False

BATCH_SIZE = 256
EPOCHS = 50

device = get_device()
plad = PLAD(2, device)
pert_optimizer = torch.optim.Adam(plad.perturbator.parameters())
clf_optimizer = torch.optim.Adam(plad.classifier.parameters())


def train():
    """
    Main training loop.

    :return: none
    """
    print(f"training on {device}")

    dataset = NormalDataset(2 ** 20, interval=(0, 10))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    if use_pretrained_classifier:
        # load pretrained classifier
        classifier = Classifier(2, device)
        classifier.load_state_dict(torch.load(PATH_PRETRAINED_CLS))

        # freeze weights of pretrained classifier
        for param in classifier.parameters():
            param.requires_grad = False

        # use the frozen pre-trained classifier in PLAD
        plad.classifier = classifier

        plotting.plotting.plot_decision("Pretrained Classifier", classifier, next(iter(train_loader)), None)

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            x = batch.to(device)
            pert_optimizer.zero_grad()
            clf_optimizer.zero_grad()

            outputs = plad(x)
            loss = L(*outputs)
            loss.backward()

            pert_optimizer.step()
            clf_optimizer.step()
            epoch_loss += loss.item()

        with torch.no_grad():
            # make a plot and calculate some performance metrics
            x_normal = next(iter(train_loader)).to(device)
            y_pred_normal, y_pred_pert, alpha, beta = plad(x_normal)
            acc_pert = torch.sum(y_pred_pert >= 0.5) / len(y_pred_pert)
            acc_normal = torch.sum(y_pred_normal <= 0.5) / len(y_pred_normal)
            print(f"EPOCH {epoch + 1}: loss={epoch_loss / BATCH_SIZE}, {acc_normal=:.2f}, {acc_pert=:.2f}")
            x_pert = alpha * x_normal + beta
            plotting.plotting.plot_decision(f"Epoch {epoch + 1}, loss={epoch_loss / BATCH_SIZE:.3f} clf_pre={use_pretrained_classifier}",
                                            plad.classifier,
                                            x_normal.cpu(), x_pert.cpu())


if __name__ == '__main__':
    train()
