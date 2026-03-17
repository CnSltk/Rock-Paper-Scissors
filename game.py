from constants import BEATS

def outcome(move1 : int, move2: int) -> tuple[int, int]:
    """
        Determine the result of one round of Rock–Paper–Scissors.

        Parameters
        ----------
        move1 : int  — Player 1's move (0=Rock, 1=Paper, 2=Scissors)
        move2 : int  — Player 2's move

        Returns
        -------
        (score1, score2) : tuple[int, int]
            +1 for a win, 0 for a draw, -1 for a loss.
            Scores always sum to 0 (zero-sum game).
    """
    if move1==move2: #draw
        return 0,0
    if BEATS[move1] == move2: #move1 beats move2
        return 1,-1
    return -1,1 #move2 beats move1