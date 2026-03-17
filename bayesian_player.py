"""
bayesian_player.py
------------------
Player 2: learns Player 1's transition matrix online using Bayesian
          updating and counters the predicted next move.

Theory
------
Each row of the unknown transition matrix is a categorical distribution.
The conjugate prior for a categorical distribution is the Dirichlet.
Starting from Dirichlet(alpha, alpha, alpha) per row, observing a
transition i → j is a closed-form update: just increment counts[i][j].
The posterior mean of row i is then counts[i] / sum(counts[i]).

Strategy: given Player 1's last move i, predict the most probable next
move as argmax of the posterior-mean row i, then play the counter-move.
"""

import numpy as np
from constants import COUNTER

class BayesianPlayer:
    """
       Maintains a Dirichlet posterior over each row of the opponent's
       transition matrix and uses it to choose a counter-move.

       Parameters
       ----------
       alpha : float
           Dirichlet prior concentration parameter for each cell.
           alpha=1 gives a uniform (non-informative) prior.
    """

    def __init__(self, alpha: float = 1.0):
        self.counts: np.ndarray = np.full((3,3), alpha, dtype=float)
        self.last_move: int | None = None

    #Bayesian Update
    def observe(self, prev_move: int, curr_move: int) -> None:
        """
             Update the posterior after observing Player 1 transition from
             prev_move to curr_move.

             Parameters
             ----------
             prev_move : int
                 Player 1's move on the previous round.
             curr_move : int
                 Player 1's move on the current round.
        """
        self.counts[prev_move][curr_move] += 1
        self.last_move = curr_move

    # Posterior mean of the learned transition matrix
    def learned_matrix(self) -> np.ndarray:
        """
             Return the posterior-mean transition matrix.

             Each row is normalised so it sums to 1 (Dirichlet posterior mean).

             Returns
             -------
             np.ndarray, shape (3, 3)
        """
        row_sums = self.counts.sum(axis=1, keepdims=True)
        return self.counts / row_sums

    # Chosse a move
    def play(self) -> int:
        """
             Predict Player 1's most likely next move from the posterior mean
             and return the counter-move that beats it.

             If no observation has been made yet, fall back to a random move.

             Returns
             -------
             int
                 Index of the chosen counter-move.
        """
        if self.last_move is None:
            return int(np.random.randint(0,3))

        probs =self.learned_matrix()[self.last_move]
        predicted_move = int(np.argmax(probs))
        return COUNTER[predicted_move]