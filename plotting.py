import numpy as np
import matplotlib.pyplot as plt

from constants import MOVES


def print_matrix(label: str, matrix: np.ndarray) -> None:
    """
    Pretty-print a 3×3 transition matrix to the console.

    Parameters
    ----------
    label  : str            Header line printed above the matrix.
    matrix : np.ndarray     Shape (3, 3) matrix to display.
    """
    print(f"\n{label}")
    print(f"{'':12s}" + "  ".join(f"{m:>10s}" for m in MOVES))
    for i, row in enumerate(matrix):
        vals = "  ".join(f"{v:10.4f}" for v in row)
        print(f"  {MOVES[i]:10s}{vals}")


def plot_scores(scores: np.ndarray, save_path: str = "rps_scores.png") -> None:
    """
    Plot the cumulative score history and the score differential.

    Parameters
    ----------
    scores    : np.ndarray, shape (n_rounds, 2)
                Cumulative scores for [Player 1, Player 2].
    save_path : str
                File path where the plot image will be saved.
    """
    rounds = np.arange(1, len(scores) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), facecolor="#0f0f14")

    # Shared axis styling
    for ax in axes:
        ax.set_facecolor("#0f0f14")
        for spine in ax.spines.values():
            spine.set_color("#2e2e3a")
        ax.tick_params(colors="#8888aa", labelsize=9)
        ax.yaxis.label.set_color("#ccccdd")
        ax.xaxis.label.set_color("#ccccdd")
        ax.grid(True, color="#1e1e2e", linewidth=0.6)

    # --- Top panel: cumulative scores ---------------------------------------
    ax1 = axes[0]
    ax1.plot(rounds, scores[:, 0], color="#ff6b6b", linewidth=1.2,
             label="Player 1 (Markov)", alpha=0.85)
    ax1.plot(rounds, scores[:, 1], color="#4ecdc4", linewidth=1.2,
             label="Player 2 (Bayesian)", alpha=0.85)
    ax1.axhline(0, color="#2e2e3a", linewidth=1)
    ax1.set_title("Cumulative Score over Rounds",
                  color="#eeeeff", fontsize=13, pad=10, fontfamily="monospace")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Score")
    ax1.legend(framealpha=0.2, labelcolor="white",
               facecolor="#1a1a22", edgecolor="#2e2e3a")

    # --- Bottom panel: score differential (P2 − P1) -------------------------
    ax2  = axes[1]
    diff = scores[:, 1] - scores[:, 0]
    ax2.fill_between(rounds, diff, 0, where=(diff >= 0),
                     color="#4ecdc4", alpha=0.35, label="Bayesian winning")
    ax2.fill_between(rounds, diff, 0, where=(diff < 0),
                     color="#ff6b6b", alpha=0.35, label="Markov winning")
    ax2.plot(rounds, diff, color="#ffffff", linewidth=0.7, alpha=0.5)
    ax2.axhline(0, color="#2e2e3a", linewidth=1)
    ax2.set_title("Score Differential  (Player 2 − Player 1)",
                  color="#eeeeff", fontsize=13, pad=10, fontfamily="monospace")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Δ Score")
    ax2.legend(framealpha=0.2, labelcolor="white",
               facecolor="#1a1a22", edgecolor="#2e2e3a")

    fig.suptitle("Rock – Paper – Scissors  ·  Bayesian Learning",
                 color="#ffffff", fontsize=15, y=1.01,
                 fontfamily="monospace", fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nPlot saved to {save_path}")