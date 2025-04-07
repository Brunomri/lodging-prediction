import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(data):
    plt.figure(figsize = (24, 12))
    sns.heatmap(data = data, cmap = 'Blues', annot = True)
    plt.show()