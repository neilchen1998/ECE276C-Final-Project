import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    data = np.load('path-PRM-samples.npy', allow_pickle=True)

    plt.scatter(data[:, 0], data[:, 1])

    plt.title('All Samples from PRM')
    plt.grid()
    plt.show()