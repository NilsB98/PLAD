import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


def plot_decision(title, model, data_normal, data_pert=None):
    # Create a grid of points covering the input space
    xmin, xmax = 0, 10
    ymin, ymax = -2, 2
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                         np.linspace(ymin, ymax, 100))
    Xgrid = np.column_stack([xx.ravel(), yy.ravel()])

    # Use the classifier to predict the class labels for the grid points
    ygrid = (model((Tensor(Xgrid)).cuda())).cpu().detach().numpy()

    # Reshape the predicted labels to match the shape of the input grid
    ygrid = ygrid.reshape(xx.shape)

    data_np = data_normal.numpy()
    # Create a heatmap-like plot of the predicted class labels
    plt.contourf(xx, yy, ygrid, cmap=plt.cm.RdBu, alpha=0.5)
    plt.scatter(data_np[:100, 0], data_np[:100, 1], c='r', label='Normal Data')
    if data_pert is not None:
        plt.scatter(data_pert[:100, 0], data_pert[:100, 1], c='b', label='Perturbed Data')
    plt.title(title)
    plt.legend()
    plt.show()
