
import matplotlib.pyplot as plt
import numpy
#plot different losses and total loss over epochs:
def plot_loss_curve(losses, title="Training Loss Curve"):

    plt.figure(figsize=(10, 6))
    losses = numpy.array(losses)
    plt.plot(losses[:, 0], label='Total Loss', color='blue')
    plt.plot(losses[:, 1], label='MSE Loss', color='orange')
    plt.plot(losses[:, 2], label='Eikonal Loss', color='green')
    if losses.shape[1] > 3:
        plt.plot(losses[:, 3], label='Curvature Loss', color='red')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)