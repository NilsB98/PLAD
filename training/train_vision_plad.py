from torchvision.transforms import Compose, ToTensor, Normalize

import utils.metrics
from data.mnist import RestrictedMNIST
from model.plad import PLAD
from utils.devices import get_device
from torch import Tensor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BATCH_SIZE = 256
EPOCHS = 50
OUTLIER_CLASSES = [0,8,9]

device = get_device()
plad = PLAD(28*28, device)
pert_optimizer = torch.optim.Adam(plad.perturbator.parameters())
clf_optimizer = torch.optim.Adam(plad.classifier.parameters())
lambd = 0.05

class TargetTransform:
    def __init__(self, outlier_classes):
        self.outlier_classes = outlier_classes

    def __call__(self, target):
        return 1 if target in self.outlier_classes else 0



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

    global lambd
    loss_fn = nn.BCELoss()

    l = loss_fn(pred_pert, torch.ones_like(pred_pert)) + lambd * (torch.norm(alpha - 1) + torch.norm(beta))
    l += loss_fn(pred_normal, torch.zeros_like(pred_normal))

    return l


transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

train_data = RestrictedMNIST('data/', download=True, transform=transform, exclude_targets=OUTLIER_CLASSES)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=64,
                                           shuffle=True)

test_data = RestrictedMNIST('data/', train=False, download=True, transform=transform, target_transform=TargetTransform(OUTLIER_CLASSES))
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=64,
                                          shuffle=True)

for epoch in range(EPOCHS):
    epoch_loss = 0
    normal_accs = []
    pert_accs = []

    for i, (imgs, targets) in enumerate(train_loader):
        x = imgs.to(device)
        pert_optimizer.zero_grad()
        clf_optimizer.zero_grad()

        pred_normal, pred_pert, alpha, beta, x_pert = plad(x)
        loss = L(pred_normal, pred_pert, alpha, beta)
        loss.backward()

        pert_optimizer.step()
        clf_optimizer.step()
        epoch_loss += loss.item()

        with torch.no_grad():
            normal_accs.append(torch.sum(pred_normal < 0.5) / len(pred_normal))
            pert_accs.append(torch.sum(pred_pert >= 0.5) / len(pred_pert))
            # show image
            if i != 0:
                continue
            x = x.cpu()
            x_pert = x_pert.cpu()
            f, axes = plt.subplots(1, 2)
            axes[0].imshow(torch.reshape(x[0], (28, 28)))
            axes[1].imshow(torch.reshape(x_pert[0], (28, 28)))
            plt.title(f'EPOCH {epoch + 1}, lambda={lambd:.2f}')
            plt.show()

    test_accs_outlier = []
    test_accs_normal = []
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(test_loader):
            x = imgs.to(device)
            pred, *_ = plad(x)

            acc_outlier = utils.metrics.bin_accuracy(targets, pred, 1)
            acc_normal = utils.metrics.bin_accuracy(targets, pred, 0)
            test_accs_outlier.append(acc_outlier)
            test_accs_normal.append(acc_normal)

    print(f"EPOCH {epoch+1}:")
    print(f"  TRAIN: acc-normal={sum(normal_accs)/len(normal_accs):.4f}, acc-pert={sum(pert_accs)/len(pert_accs):.4f}")
    print(f"  TEST: acc-normal={sum(test_accs_normal)/len(test_accs_normal):.4f}, acc-outlier={sum(test_accs_outlier)/len(test_accs_outlier):.4f}")
    lambd += .01
