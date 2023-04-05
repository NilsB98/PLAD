import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import plotting.plotting
from data.function_ds import NormalDataset
from data.sphere import SphereDataset, RectangleDataset
from model.plad import PLAD, Classifier
from utils.devices import get_device


def L(pred_normal, pred_pert, points_normal, points_pert) -> Tensor:
    """
    Loss used to optimize PLAD with:
    y_true = 0
    y_pert = 1

    :param points_normal:
    :param points_pert:
    :param pred_normal: predictions for the normal data points
    :param pred_pert: predictions for the perturbed data points
    :param alpha: alpha calculated by the generator
    :param beta: beta calculated by the generator
    :return:
    """

    loss_fn = nn.BCELoss()

    # return lambd * torch.norm(points_normal - points_pert, 2)
    l = loss_fn(pred_pert, torch.ones_like(pred_pert)) + lambd * torch.norm(points_normal - points_pert, 2)

    if not use_pretrained_classifier:
        l += loss_fn(pred_normal, torch.zeros_like(pred_normal))

    return l


PATH_PRETRAINED_CLS = r'../checkpoints/classifier.pt'
PATH_PLAD = r'../checkpoints/plad.pt'
use_pretrained_classifier = False
use_pretrained_plad = False

BATCH_SIZE = 512
EPOCHS = 150
lambd = 0.01

device = get_device()
plad = PLAD(2, device)
pert_optimizer = torch.optim.Adam(plad.perturbator.parameters())
clf_optimizer = torch.optim.Adam(plad.classifier.parameters())


def train():
    """
    Main training loop.

    :return: none
    """
    global lambd
    print(f"training on {device}")

    # dataset = NormalDataset(2 ** 21, interval=(0, 10))
    # dataset.normalize()
    # dataset = SphereDataset(2**21, {'rad': 1})
    dataset = RectangleDataset(2 ** 21, {'width': 1, 'height': 2})
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

    if use_pretrained_plad:
        plad.load_state_dict(torch.load(PATH_PLAD))

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            x_normal: Tensor = batch.to(device)
            x_normal.requires_grad = True
            pert_optimizer.zero_grad()
            if not use_pretrained_classifier:
                clf_optimizer.zero_grad()

            y_pred_normal, y_pred_pert, x_pert = plad(x_normal)
            loss = L(y_pred_normal, y_pred_pert, x_normal, x_pert)
            loss.backward()

            pert_optimizer.step()
            if not use_pretrained_classifier:  #  and i % 5 == 0
                clf_optimizer.step()
            epoch_loss += loss.item()

        with torch.no_grad():
            # make a plot and calculate some performance metrics
            x_normal = next(iter(train_loader)).to(device)
            y_pred_normal, y_pred_pert, x_pert = plad(x_normal)
            acc_pert = torch.sum(y_pred_pert >= 0.5) / len(y_pred_pert)
            acc_normal = torch.sum(y_pred_normal <= 0.5) / len(y_pred_normal)
            print(f"EPOCH {epoch + 1}: loss={epoch_loss / BATCH_SIZE}, {acc_normal=:.2f}, {acc_pert=:.2f}")
            plotting.plotting.plot_decision(f"Epoch {epoch + 1}, loss={epoch_loss / BATCH_SIZE:.3f} lamd={lambd:.2f}",
                                            plad.classifier,
                                            x_normal.cpu(), x_pert.cpu())

        # torch.save(plad.state_dict(), PATH_PLAD)
        # lambd += .01

if __name__ == '__main__':
    train()
