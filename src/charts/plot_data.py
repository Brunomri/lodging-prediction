import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(data):
    plt.figure(figsize = (24, 12))
    sns.heatmap(data = data, cmap = 'Blues', annot = True)
    plt.show()

def plot_scatter(x, y):
    plt.figure(figsize = (24, 12))
    sns.scatterplot(x = x, y = y)
    plt.show()

def plot_histogram(data, column):
    plt.figure(figsize = (24, 12))
    sns.histplot(data = data, x = column)
    plt.show()