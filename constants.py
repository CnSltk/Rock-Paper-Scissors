"""
constants.py
------------
Shared game constants used across all modules.
"""
# Move names indexed 0, 1, 2
MOVES = ['Rock', 'Paper', 'Scissors']

# BEATS[x] = y  means  move x  beats  move y
#   Rock(0)     beats Scissors(2)
#   Paper(1)    beats Rock(0)
#   Scissors(2) beats Paper(1)
BEATS = {0:2, 1:0 , 2:1}

# COUNTER[x] = y  means  y beats x  (the winning response to move x)
#   Counter to Rock(0)     → Paper(1)
#   Counter to Paper(1)    → Scissors(2)
#   Counter to Scissors(2) → Rock(0)
COUNTER = {0:1, 1:2 , 2:0}

# Simulation settings
N_ROUNDS= 1000
SEED = 42