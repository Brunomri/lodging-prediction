import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(data):
    plt.figure(figsize = (24, 12))
    sns.heatmap(data = data, cmap = 'Blues', annot = True)
    plt.show()

def plot_scatter(pred, test):
    plt.figure(figsize = (24, 12))
    sns.scatterplot(x = pred, y = test)
    plt.show()

def plot_distribution(df):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    df['realSum'].hist(bins=50)

    plt.subplot(1, 2, 2)
    df.boxplot(column='realSum')
    plt.show()