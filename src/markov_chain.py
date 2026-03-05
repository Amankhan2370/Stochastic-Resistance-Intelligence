"""
Markov Chain modeling for antimicrobial resistance states.
Computes co-occurrence statistics and builds the transition matrix.
"""

import os
import numpy as np
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the antimicrobial resistance dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def compute_co_occurrence_counts(df: pd.DataFrame) -> dict:
    """
    Compute antimicrobial co-occurrence statistics using numpy logical operations.

    amp_pen:  Number of records where Ampicillin == 1 AND Penicillin == 1
    amp_nmdr: Number of records where Ampicillin == 1 AND Not_MDR == 1
    pen_nmdr: Number of records where Penicillin == 1 AND Not_MDR == 1

    Args:
        df: DataFrame with Ampicillin, Penicillin, Not_MDR columns.

    Returns:
        Dictionary with amp_pen, amp_nmdr, pen_nmdr counts.
    """
    amp = np.array(df["Ampicillin"])
    pen = np.array(df["Penicillin"])
    nmdr = np.array(df["Not_MDR"])

    # Use numpy logical operations for co-occurrence
    amp_pen = np.sum((amp == 1) & (pen == 1))
    amp_nmdr = np.sum((amp == 1) & (nmdr == 1))
    pen_nmdr = np.sum((pen == 1) & (nmdr == 1))

    return {"amp_pen": amp_pen, "amp_nmdr": amp_nmdr, "pen_nmdr": pen_nmdr}


def build_transition_matrix(counts: dict) -> np.ndarray:
    """
    Build the Markov chain transition matrix.

    States: 0 = Ampicillin, 1 = Penicillin, 2 = Not_MDR

    T[i][j] = probability of transitioning from state i to state j.

    Args:
        counts: Dictionary with amp_pen, amp_nmdr, pen_nmdr.

    Returns:
        3x3 transition matrix as NumPy array.
    """
    amp_pen = counts["amp_pen"]
    amp_nmdr = counts["amp_nmdr"]
    pen_nmdr = counts["pen_nmdr"]

    # Avoid division by zero by using a small epsilon
    eps = 1e-10
    denom_amp = amp_nmdr + amp_pen + eps
    denom_pen = pen_nmdr + amp_pen + eps
    denom_nmdr = amp_nmdr + pen_nmdr + eps

    # Transition matrix as specified
    # Row 0 (Ampicillin): transitions to Penicillin, Not_MDR
    # Row 1 (Penicillin): transitions to Ampicillin, Not_MDR
    # Row 2 (Not_MDR): transitions to Ampicillin, Penicillin
    T = np.array(
        [
            [0, amp_pen / denom_amp, amp_nmdr / denom_amp],
            [amp_pen / denom_pen, 0, pen_nmdr / denom_pen],
            [amp_nmdr / denom_nmdr, pen_nmdr / denom_nmdr, 0],
        ],
        dtype=float,
    )

    return T


def run_markov_chain(data_path: str = "data/amr_ds.csv") -> tuple:
    """
    Main entry point: load data, compute counts, build transition matrix.

    Args:
        data_path: Relative path to dataset from project root.

    Returns:
        Tuple of (counts dict, transition matrix) for use by other modules.
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(script_dir, data_path)

    df = load_dataset(full_path)
    counts = compute_co_occurrence_counts(df)
    T = build_transition_matrix(counts)

    print("\n" + "=" * 60)
    print("PART 2 & 3: Antimicrobial Co-occurrence & Markov Chain")
    print("=" * 60)
    print("\nCo-occurrence counts (numpy logical operations):")
    print(f"  amp_pen:  {counts['amp_pen']}  (Ampicillin=1 AND Penicillin=1)")
    print(f"  amp_nmdr: {counts['amp_nmdr']} (Ampicillin=1 AND Not_MDR=1)")
    print(f"  pen_nmdr: {counts['pen_nmdr']} (Penicillin=1 AND Not_MDR=1)")
    print("\nTransition matrix T (states: Ampicillin, Penicillin, Not_MDR):")
    print(T)
    print()

    return counts, T
