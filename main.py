from constants import N_ROUNDS
from simulation import run
from plotting import print_matrix,plot_scores

def main():
    scores,bayesian_player,true_T = run()
    print("=" * 55)
    print(f"  Simulation complete  ({N_ROUNDS} rounds)")
    print("=" * 55)
    print(f"  Player 1 (Markov)   final score: {scores[-1, 0]:+d}")
    print(f"  Player 2 (Bayesian) final score: {scores[-1, 1]:+d}")
    print_matrix("True transition matrix (Player 1):", true_T)
    print_matrix("Learned transition matrix (Player 2 posterior mean):",
                 bayesian_player.learned_matrix())
    # --- Plot ---
    plot_scores(scores, save_path="rps_scores.png")

if __name__ == "__main__":
    main()