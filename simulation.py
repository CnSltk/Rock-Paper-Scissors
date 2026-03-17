import numpy as np

from constants import N_ROUNDS,SEED
from markov_player import MarkovPlayer
from bayesian_player import BayesianPlayer
from game import outcome

def run(n_rounds: int= N_ROUNDS, seed: int= SEED):
    """
       Simulate a full game between the Markov player and the Bayesian player.

       Parameters
       ----------
       n_rounds : int   Number of rounds to play.
       seed     : int   Random seed for reproducibility.

       Returns
       -------
       scores         : np.ndarray, shape (n_rounds, 2)
                        Cumulative scores [player1, player2] after each round.
       bayesian_player: BayesianPlayer
                        The trained player (holds the learned transition matrix).
       true_T         : np.ndarray, shape (3, 3)
                        The true transition matrix used by Player 1.
    """
    rng=np.random.default_rng(seed)

    # True transition matrix — non-uniform so there is a pattern to learn
    true_T= np.array([
        [0.1,0.6,0.3],  # after Rock:tends to play Paper
        [0.3,0.1,0.6],  # after Paper:tends to play Scissors
        [0.6,0.3,0.1]   # after Scissors:tends to play Rock
    ])
    p1=MarkovPlayer(true_T,rng)
    p2=BayesianPlayer(alpha=1.0)
    scores=np.zeros((n_rounds,2),dtype=int)

    # Track Player 1's previous move for the Bayesian update
    prev_p1_move=p1.last_move
    cumulative = [0,0]

    for t in range(n_rounds):
        move1=p1.play()
        move2=p2.play()
        s1,s2 = outcome(move1,move2)
        cumulative[0] += s1
        cumulative[1] += s2
        scores[t]=cumulative
        p2.observe(prev_p1_move,move1)
        prev_p1_move = move1
    return scores,p2,true_T