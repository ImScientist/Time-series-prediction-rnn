import matplotlib.pyplot as plt


def plot_predictions(time, y, y_hat, t_train, idx):
    fig = plt.figure(figsize=(20, 5))

    plt.plot(time, y, linestyle='solid', label=f'y_{idx}')
    plt.plot(time, y_hat, linestyle='dashed', label=f'y_hat_{idx}')
    plt.axvline(x=t_train, ymin=min(y), ymax=max(y), color='grey', linestyle='--')

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(fontsize=14)
    plt.grid(True)

    return fig
