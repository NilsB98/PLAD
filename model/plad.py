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

        self.fc1 = nn.Linear(dim, 100, device=device)
        self.fc1_1 = nn.Linear(100, 100, device=device)
        self.fc1_2 = nn.Linear(100, 2 * dim, device=device)
        self.scaling = nn.Linear(2 * dim, 2 * dim, device=device)

    def forward(self, x):
        """
        Returns the factors alpha and beta.
        They are stacked in the returned tensor, i.e. the first half of params belongs to alpha
        and the second half to beta.

        :param x: Input sample to perturb
        :return: alpha, beta contained in a Tensor.
        """
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc1_1(x))
        x = F.tanh(self.fc1_2(x))
        x = self.scaling(x)

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
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Make a forward pass which will classify the normal data x, as well as generate and classify perturbed data
        based on x.

        :param x: normal data
        :return: prediction(x_pert), prediction(x), alpha, beta
        """

        x = self.flatten(x)
        pert_params = self.perturbator(x)
        alpha = pert_params[:, self.dim:]
        beta = pert_params[:, :self.dim]

        #
        x_pert = alpha * x + beta

        return self.classifier(x), self.classifier(x_pert), alpha, beta, x_pert