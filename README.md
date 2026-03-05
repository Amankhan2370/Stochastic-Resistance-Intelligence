<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/numpy-1.20+-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

<h1 align="center">Antimicrobial Resistance Probabilistic Modeling</h1>
<p align="center">
  <i>Machine learning & probabilistic modeling for AMR data</i>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Example Output](#example-output)

---

## Overview

This project analyzes **antimicrobial resistance (AMR)** data using a combination of machine learning and probabilistic models. It demonstrates:

| Component | Technique | Purpose |
|-----------|-----------|---------|
| Classification | Gaussian Naïve Bayes | Predict Not_MDR status from resistance features |
| State modeling | Markov chain | Model transitions between resistance states |
| Long-run behavior | Eigenvector decomposition | Compute stationary distribution π |
| Sequence inference | Viterbi algorithm | Decode hidden states from infection observations |

---

## Features

- **Naïve Bayes classifier** — Predicts `Not_MDR` from `Ampicillin` and `Penicillin` (75/25 train-test split)
- **Co-occurrence analysis** — Computes `amp_pen`, `amp_nmdr`, `pen_nmdr` using NumPy logical ops
- **3-state Markov chain** — Transition matrix between Ampicillin, Penicillin, and Not_MDR states
- **Stationary distribution** — π such that πT = π, via eigenvector decomposition
- **Hidden state inference** — Viterbi-style decoding for infection observation sequences

---

## Quick Start

```bash
# Clone and enter
cd antimicrobial-resistance-probabilistic-modeling

# Install
pip install -r requirements.txt

# Run
python main.py
```

<details>
<summary><b>Optional: use a virtual environment</b></summary>

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
python main.py
```

</details>

---

## Dataset

| Column | Type | Description |
|--------|------|-------------|
| `Ampicillin` | 0/1 | Resistance to Ampicillin |
| `Penicillin` | 0/1 | Resistance to Penicillin |
| `Not_MDR` | 0/1 | Not Multi-Drug Resistant (target) |

**Location:** `data/amr_ds.csv`

---

## Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│   Load AMR      │────▶│  Naïve Bayes     │     │  Co-occurrence counts   │
│   Dataset       │     │  Classification  │     │  amp_pen, amp_nmdr,     │
└────────┬────────┘     └──────────────────┘     │  pen_nmdr               │
         │                                        └───────────┬─────────────┘
         │                                                    │
         │                                                    ▼
         │                                        ┌─────────────────────────┐
         │                                        │  Markov transition      │
         │                                        │  matrix T               │
         │                                        └───────────┬─────────────┘
         │                                                    │
         │                                                    ▼
         │                                        ┌─────────────────────────┐
         │                                        │  Stationary             │
         │                                        │  distribution π         │
         │                                        └─────────────────────────┘
         │
         └──────────────────────────────────────▶┌─────────────────────────┐
                                                 │  Hidden state inference │
                                                 │  (Viterbi)              │
                                                 └─────────────────────────┘
```

---

## Project Structure

```
antimicrobial-resistance-probabilistic-modeling/
├── data/
│   └── amr_ds.csv              # Binary AMR dataset
├── src/
│   ├── naive_bayes_model.py    # Part 1: Classification
│   ├── markov_chain.py         # Part 2–3: Co-occurrence & transition matrix
│   ├── stationary_distribution.py  # Part 4: Eigenvector-based π
│   └── hidden_state_prediction.py  # Part 5: Viterbi decoding
├── main.py                     # Run full pipeline
├── requirements.txt
├── README.md
└── results/
```

---

## Example Output

```
############################################################
# Antimicrobial Resistance - Probabilistic Modeling
############################################################

============================================================
PART 1: Naïve Bayes Classification
============================================================
Training size: 272
Testing size:  91
Accuracy:      0.9451

============================================================
PART 2 & 3: Antimicrobial Co-occurrence & Markov Chain
============================================================

Co-occurrence counts (numpy logical operations):
  amp_pen:  107  (Ampicillin=1 AND Penicillin=1)
  amp_nmdr: 6    (Ampicillin=1 AND Not_MDR=1)
  pen_nmdr: 55   (Penicillin=1 AND Not_MDR=1)

Transition matrix T (states: Ampicillin, Penicillin, Not_MDR):
[[0.         0.9469  0.0531]
 [0.6605    0.      0.3395]
 [0.0984    0.9016  0.    ]]

============================================================
PART 4: Stationary Distribution
============================================================

Stationary distribution (πT = π):
  π(Ampicillin): 0.3363
  π(Penicillin): 0.4821
  π(Not_MDR):    0.1815

============================================================
PART 5: Hidden State Inference
============================================================

Observed: [Infection, No Infection, Infection]
Most probable resistance state sequence: ['Pen', 'NMDR', 'Pen']
```

---

## Requirements

| Package | Role |
|---------|------|
| `numpy` | Linear algebra, array ops |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Naïve Bayes, train-test split, metrics |

---

## License

MIT
