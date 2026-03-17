"""
markov_player.py
----------------
Player 1: follows a fixed Markov chain strategy.

The transition matrix T is a 3×3 row-stochastic matrix where T[i][j]
is the probability of playing move j given that the previous move was i.
This means each row must sum to exactly 1.
"""

import numpy as np

class MarkovPlayer:
    """
        Chooses the next move by sampling from the row of the transition
        matrix that corresponds to the previous move played.

        Parameters
        ----------
        transition_matrix : np.ndarray, shape (3, 3)
            Row-stochastic matrix defining the Markov strategy.
        rng : np.random.Generator
            Random number generator (passed in for reproducibility).
        """
    def __init__(self,transition_matrix: np.ndarray, rng: np.random.Generator):
        assert transition_matrix.shape == (3, 3)
        assert np.allclose(transition_matrix.sum(axis=1),1.0), \
            "Every row must sum to 1."
        self.T = transition_matrix
        self.rng = rng

        self.last_move: int = int(rng.integers(0,3))

    def play(self) -> int:
        """
        Sample the next move from T[last_move], then update last_move.

        Returns
        int
            Index of the chosen move (0=Rock, 1=Paper, 2=Scissors).
        """
        probs= self.T[self.last_move]
        move = int(self.rng.choice(3, p=probs))
        self.last_move = move
        return move


