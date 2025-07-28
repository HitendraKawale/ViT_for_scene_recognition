import os
import csv
import matplotlib.pyplot as plt

RESULTS_DIR = "results/runs"

# Colors for distinct plots
colors = ["blue", "green", "red", "orange", "purple", "cyan", "magenta", "brown"]

plt.figure(figsize=(12, 6))

for i, run_dir in enumerate(sorted(os.listdir(RESULTS_DIR))):
    run_path = os.path.join(RESULTS_DIR, run_dir, "metrics.csv")
    if not os.path.exists(run_path):
        continue

    epochs, train_acc, val_acc = [], [], []

    with open(run_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    color = colors[i % len(colors)]
    plt.plot(epochs, train_acc, label=f"{run_dir} - Train", linestyle="--", color=color)
    plt.plot(epochs, val_acc, label=f"{run_dir} - Val", linestyle="-", color=color)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy per Run")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plot_accuracy.png")
plt.show()

