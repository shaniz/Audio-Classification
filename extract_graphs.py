import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_vs_layer(d, prefix):
    plt.figure(figsize=(10, 6))

    # Plot Accuracy and Best Accuracy for each layer
    plt.plot(df['Layer'], df['Acc'], label='Accuracy', marker='o', linestyle='-', color='blue')
    plt.plot(df['Layer'], df['Best Acc'], label='Best Accuracy', marker='o', linestyle='--', color='red')

    plt.xlabel(f'{prefix} Weights Till')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Accuracy vs Layer for {df["Model"][0]} on {df["Dataset"][0]} (Fold {df["Fold"][0]})')  # Same for all rows in the csv
    plt.legend()
    plt.show()


files = {'Pretrained': 'results/B3/B3-densenet-fusion.csv', 'Freeze': 'results/B3/B3-densenet-freeze.csv'}

for prefix, f in files.items():
    df = pd.read_csv(f)
    plot_accuracy_vs_layer(df, prefix)
