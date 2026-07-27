import os
import matplotlib.pyplot as plt

plt.ion()


def plot(scores, mean_scores):
    """Plots the current score and the running average score of the AI, then saves it as a PNG."""
    plt.clf()
    plt.title("Training Progress...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    # Plot both raw scores and the calculated mean scores
    plt.plot(scores, label="Current Score")
    plt.plot(mean_scores, label="Running Average")
    plt.ylim(ymin=0)

    # Place text labels at the end of the lines on the plot
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    plt.show()
    plt.pause(0.1)

    plot_folder_path = "./data/plots"
    os.makedirs(plot_folder_path, exist_ok=True)
    file_path = os.path.join(plot_folder_path, "progress.png")
    plt.savefig(file_path)