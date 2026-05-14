r"""
Visual evaluation script for the CNN-DQN dynamic traffic agent.

Loads the trained CNN-DQN model from
`artifacts/dqn/exp4_cnn_dqn_dynamic/models/exp4_dqn_cnn.pt`
(produced by `src/deep_learning/exp4_cnn_dqn_dynamic.py`) and renders
10 episodes using the highway-env Pygame visualizer.

Usage:
    1. First, train:   python .\src\deep_learning\exp4_cnn_dqn_dynamic.py
    2. Then, watch:    python .\src\deep_learning\test_exp4_visualization.py
"""

import os
import sys

# Make sure local imports work when running from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from exp4_cnn_dqn_dynamic import (
    evaluate,
    MODEL_PATH,
)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No trained model found at '{MODEL_PATH}'.")
        print("       Run  python src/exp4_cnn_dqn_dynamic.py  first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH} ...")

    # Visual evaluation: 10 rendered episodes (Pygame window)
    print("--- Visual evaluation (10 episodes) ---")
    evaluate(episodes=10)


if __name__ == "__main__":
    main()
