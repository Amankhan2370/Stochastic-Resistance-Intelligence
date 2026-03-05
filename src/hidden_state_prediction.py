"""
Hidden state inference using a simplified Viterbi-like algorithm.
Infers the most probable resistance state sequence from infection observations.
"""

import numpy as np

# State indices: 0 = Amp (Ampicillin), 1 = Pen (Penicillin), 2 = NMDR (Not_MDR)
STATE_NAMES = ["Amp", "Pen", "NMDR"]

# Observed sequence: Infection=1, No Infection=0
OBSERVED = [1, 0, 1]  # [Infection, No Infection, Infection]

# Emission probabilities: P(Observation | State)
# Columns: No Infection (0), Infection (1)
# Rows: Amp, Pen, NMDR
EMISSION = np.array(
    [
        [0.4, 0.6],  # Amp: P(No Inf|Amp)=0.4, P(Inf|Amp)=0.6
        [0.3, 0.7],  # Pen: P(No Inf|Pen)=0.3, P(Inf|Pen)=0.7
        [0.8, 0.2],  # NMDR: P(No Inf|NMDR)=0.8, P(Inf|NMDR)=0.2
    ]
)

# Equal starting probability for each state
INITIAL_PROBS = np.array([1 / 3, 1 / 3, 1 / 3])

# Transition probabilities: uniform transitions between different states
# (simplified model - equal probability of going to other states)
NUM_STATES = 3
TRANSITION = np.array(
    [
        [0, 0.5, 0.5],  # From Amp: to Pen or NMDR
        [0.5, 0, 0.5],  # From Pen: to Amp or NMDR
        [0.5, 0.5, 0],  # From NMDR: to Amp or Pen
    ]
)


def viterbi(
    observations: list,
    initial_probs: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> list:
    """
    Simplified Viterbi algorithm to find the most probable hidden state sequence.

    Args:
        observations: List of observed values (0=No Infection, 1=Infection).
        initial_probs: Initial state probabilities.
        transition: State transition matrix T[i][j] = P(s_j | s_i).
        emission: Emission matrix E[s][o] = P(obs=o | state=s).

    Returns:
        List of state indices representing the most probable path.
    """
    T = len(observations)
    n_states = len(initial_probs)

    # Viterbi matrices
    viterbi_probs = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=int)

    # Initialize with first observation
    for s in range(n_states):
        obs_idx = int(observations[0])
        viterbi_probs[s, 0] = initial_probs[s] * emission[s, obs_idx]

    # Forward pass
    for t in range(1, T):
        obs_idx = int(observations[t])
        for s in range(n_states):
            # Max over previous states
            trans_probs = viterbi_probs[:, t - 1] * transition[:, s]
            best_prev = np.argmax(trans_probs)
            viterbi_probs[s, t] = trans_probs[best_prev] * emission[s, obs_idx]
            backpointer[s, t] = best_prev

    # Backtrack to find best path
    path = [0] * T
    path[T - 1] = np.argmax(viterbi_probs[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    return path


def run_hidden_state_prediction() -> list:
    """
    Main entry point: run Viterbi on infection observations and print result.

    Returns:
        List of state names for the most probable sequence.
    """
    path_indices = viterbi(OBSERVED, INITIAL_PROBS, TRANSITION, EMISSION)
    path_names = [STATE_NAMES[i] for i in path_indices]

    print("\n" + "=" * 60)
    print("PART 5: Hidden State Inference")
    print("=" * 60)
    print("\nObserved sequence: [Infection, No Infection, Infection]")
    print("\nEmission probabilities:")
    print("  State      No Infection   Infection")
    print("  Amp        0.4            0.6")
    print("  Pen        0.3            0.7")
    print("  NMDR       0.8            0.2")
    print("\nMost probable resistance state sequence:")
    print(f"  {path_names}")
    print()

    return path_names
