import torch
import torch.nn as nn
import torch.nn.functional as F


class Perturbator(nn.Module):
    """
    Implementation of the PLAD model.
    """
    def __init__(self, dim, device):
        super(Perturbator, self).__init__()
        self.device = device

        self.fc1 = nn.Linear(dim, 50, device=device)
        self.fc2 = nn.Linear(50, 2 * dim, device=device)

    def forward(self, x):
        """
        Returns the factors alpha and beta.
        They are stacked in the returned tensor, i.e. the first half of params belongs to alpha
        and the second half to beta.

        :param x: Input sample to perturb
        :return: alpha, beta contained in a Tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class Classifier(nn.Module):
    def __init__(self, dim, device):
        super(Classifier, self).__init__()

        self.device = device

        self.fc1 = nn.Linear(dim, 100, device=device)
        self.fc2 = nn.Linear(100, 1, device=device)

    def forward(self, x):
        """
        Classify a given sample in a binary classification with:
        0 -> normal data
        1 -> abnormal (perturbed data)

        :param x: sample to classify
        :return: probability to be abnormal.
        """
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class PLAD(nn.Module):
    def __init__(self, dim, device):
        super().__init__()

        self.dim = dim

        self.perturbator = Perturbator(dim, device)
        self.classifier = Classifier(dim, device)

    def forward(self, x):
        """
        Make a forward pass which will classify the normal data x, as well as generate and classify perturbed data
        based on x.

        :param x: normal data
        :return: prediction(x_pert), prediction(x), alpha, beta
        """

        pert_params = self.perturbator(x)
        alpha = pert_params[:, self.dim:]
        beta = pert_params[:, :self.dim]

        x_pert = alpha * x + beta

        return self.classifier(x), self.classifier(x_pert), alpha, beta


class OptimalClassifier(nn.Module):
    """
    Classifier to optimally classify data points for the sinusoid toy problem.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, X):
        epsilon = .5
        is_in_normal_area = (torch.sin(X[:, 0]) - epsilon < X[:, 1]) & (X[:, 1] < torch.sin(X[:, 0]) + epsilon)

        return torch.logical_not(is_in_normal_area).int().float()