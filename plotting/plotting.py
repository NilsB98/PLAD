import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


def plot_decision(title, model, data_normal: Tensor, data_pert: Tensor = None, num_points=100):
    # Create a grid of points covering the input space
    mins = Tensor.min(data_normal, dim=0).values
    maxs = Tensor.max(data_normal, dim=0).values

    xx, yy = np.meshgrid(np.linspace(mins[0] - 1, maxs[0] + 1, 100),
                         np.linspace(mins[1] - 1, maxs[1] + 1, 100))
    Xgrid = np.column_stack([xx.ravel(), yy.ravel()])

    # Use the classifier to predict the class labels for the grid points
    ygrid = (model((Tensor(Xgrid)).cuda())).cpu().detach().numpy()

    # Reshape the predicted labels to match the shape of the input grid
    ygrid = ygrid.reshape(xx.shape)

    data_np = data_normal.numpy()
    # Create a heatmap-like plot of the predicted class labels
    levels = [0, 0.4, 0.5, 0.6, 1]
    plt.contourf(xx, yy, ygrid, cmap=plt.cm.RdBu, alpha=0.5, vmin=0, vmax=1, levels=levels)
    plt.scatter(data_np[:num_points, 0], data_np[:num_points, 1], c='r', label='Normal Data')
    if data_pert is not None:
        plt.scatter(data_pert[:num_points, 0], data_pert[:num_points, 1], c='b', label='Perturbed Data')
        # draw lines between the pairs of normal and perturbed points:
        for x_normal, x_pert in zip(data_normal[:num_points], data_pert[:num_points]):
            plt.plot([x_normal[0], x_pert[0]], [x_normal[1], x_pert[1]], c='grey', alpha=.7)
    plt.title(title)
    plt.legend()
    plt.show()
