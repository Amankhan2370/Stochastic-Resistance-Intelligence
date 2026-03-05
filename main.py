"""
Main entry point for antimicrobial resistance probabilistic modeling.
Orchestrates all analysis components and prints results.
"""

from src.naive_bayes_model import run_naive_bayes
from src.markov_chain import run_markov_chain
from src.stationary_distribution import run_stationary_distribution
from src.hidden_state_prediction import run_hidden_state_prediction


def main():
    """Run the complete antimicrobial resistance analysis pipeline."""
    print("\n" + "#" * 60)
    print("# Antimicrobial Resistance - Probabilistic Modeling")
    print("#" * 60)

    # Part 1: Naïve Bayes classification
    run_naive_bayes(data_path="data/amr_ds.csv")

    # Part 2 & 3: Co-occurrence statistics and Markov chain
    counts, transition_matrix = run_markov_chain(data_path="data/amr_ds.csv")

    # Part 4: Stationary distribution
    run_stationary_distribution(T=transition_matrix)

    # Part 5: Hidden state inference
    run_hidden_state_prediction()

    print("=" * 60)
    print("Analysis complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
