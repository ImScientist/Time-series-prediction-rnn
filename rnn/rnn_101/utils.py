import matplotlib.pyplot as plt


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def plot_predictions(time, y, y_pred, t_warmup, t_train):
    fig = plt.figure(figsize=(10, 5))

    plot_series(
        time, y, format="-", start=0, end=None, label='y')
    plot_series(
        time[t_warmup:], y_pred[t_warmup:], format=".", start=0, end=None,
        label='y_hat')
    plt.axvline(
        x=t_train, ymin=min(y), ymax=max(y), linestyle='--', color='grey')

    return fig
