import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_heatmap(data, title=""):
    plt.figure(figsize = (24, 12))
    sns.heatmap(data = data, cmap = 'Blues', annot = True)
    plt.title(title)
    plt.show()

def plot_scatter(pred, test, title="", x_label="predicted", y_label="real"):
    plt.figure(figsize = (24, 12))
    sns.scatterplot(x = pred, y = test)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    max_val = max(max(pred), max(test))
    step = 100
    ticks = np.arange(0, max_val + step, step)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.plot([0, max_val], [0, max_val], 'r--')

    plt.show()

def plot_distribution(df, col, title=""):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    df[col].hist(bins=50)

    plt.subplot(1, 2, 2)
    df.boxplot(column=col)
    plt.suptitle(title)
    plt.show()