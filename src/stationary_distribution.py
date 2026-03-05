"""
Stationary distribution computation for the antimicrobial resistance Markov chain.
Uses eigenvector decomposition: πT = π means π is left eigenvector of T for eigenvalue 1.
"""

import numpy as np


def compute_stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution π such that πT = π.

    For a transition matrix T, the stationary distribution is the left eigenvector
    corresponding to eigenvalue 1. Equivalently, it is the right eigenvector of T^T.

    Steps:
    1. Compute eigenvalues and eigenvectors of T^T (transpose)
    2. Find eigenvector corresponding to eigenvalue 1 (or closest to 1)
    3. Normalize so the sum equals 1

    Args:
        T: 3x3 transition matrix.

    Returns:
        Normalized stationary distribution [π_Amp, π_Pen, π_NMDR].
    """
    # Compute eigenvalues and eigenvectors of T^T
    eigenvalues, eigenvectors = np.linalg.eig(T.T)

    # Find index of eigenvalue closest to 1 (real part)
    idx = np.argmin(np.abs(eigenvalues - 1.0))

    # Extract corresponding eigenvector (column)
    stationary = np.real(eigenvectors[:, idx])

    # Ensure non-negative (stationary distribution must be a probability vector)
    stationary = np.abs(stationary)

    # Normalize so sum equals 1
    stationary = stationary / stationary.sum()

    return stationary


def run_stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Main entry point: compute and print stationary distribution.

    Args:
        T: Transition matrix from markov_chain module.

    Returns:
        Stationary distribution array [π_Amp, π_Pen, π_NMDR].
    """
    pi = compute_stationary_distribution(T)

    print("\n" + "=" * 60)
    print("PART 4: Stationary Distribution")
    print("=" * 60)
    print("\nStationary distribution (πT = π, computed via eigenvector decomposition):")
    print(f"  π(Ampicillin): {pi[0]:.4f}")
    print(f"  π(Penicillin): {pi[1]:.4f}")
    print(f"  π(Not_MDR):    {pi[2]:.4f}")
    print(f"  Sum:           {pi.sum():.4f}")
    print()

    return pi
