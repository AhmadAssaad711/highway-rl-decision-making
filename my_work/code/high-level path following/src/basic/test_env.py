"""
Test / Visual Evaluation Script
─────────────────────────────────
Loads the trained Q-table from  artifacts/basic/q_table.pkl  (produced by
static_obstacle_avoidance.py) and renders 10 episodes using the
highway-env Pygame visualiser.

Usage:
    1. First, train:   python src/static_obstacle_avoidance.py
    2. Then, watch:    python src/test_env.py
"""

import os
import sys

# ── Make sure 'src/' imports work when running from the project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from static_obstacle_avoidance import load_q_table, evaluate, Q_TABLE_PATH


def main():
    # Load the Q-table saved during training
    if not os.path.exists(Q_TABLE_PATH):
        print(f"ERROR: No Q-table found at '{Q_TABLE_PATH}'.")
        print("       Run  python src/static_obstacle_avoidance.py  first.")
        sys.exit(1)

    print(f"Loading Q-table from {Q_TABLE_PATH} ...")
    q_table = load_q_table()
    print(f"  {len(q_table)} states loaded.\n")

    # Visual evaluation — 10 rendered episodes
    print("--- Visual evaluation (10 episodes) ---")
    evaluate(q_table, episodes=10)


if __name__ == "__main__":
    main()
