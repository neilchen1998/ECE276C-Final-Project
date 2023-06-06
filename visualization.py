import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    data = np.load('path-PRMStar.npy', allow_pickle=True)

    plt.figure(figsize=[5, 5])
    ax = plt.axes([0.1, 0.1, 0.8, 0.8], xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi))
    points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels
    radius = 0.25
    points_radius = 2 * radius / 1.0 * points_whole_ax
    ax.scatter(0.5, 0.5, s=points_radius**2, color='gray')

    for i, point in enumerate(data):
        ax.scatter(point[0], point[1])
        if i > 0:
            x_values = [data[i-1][0], data[i][0]]
            y_values = [data[i-1][1], data[i][1]]
            plt.plot(x_values, y_values, linestyle="--")

    plt.title('PRMStar')
    plt.grid()
    plt.show()