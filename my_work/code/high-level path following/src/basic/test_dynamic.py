"""
Test / Visual Evaluation Script for Dynamic Traffic Avoidance
──────────────────────────────────────────────────────────────
Loads the trained Q-table from  artifacts/basic/dynamic_q_table.pkl  (produced by
dynamic_obstacle_v_cnst.py) and renders 10 episodes using the
highway-env Pygame visualiser.

Usage:
    1. First, train:   python src/dynamic_obstacle_v_cnst.py
    2. Then, watch:    python src/test_dynamic.py
"""

import os
import sys

# ── Make sure 'src/' imports work when running from the project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dynamic_obstacle_v_cnst import (
    load_q_table,
    evaluate,
    compute_decision_accuracy,
    Q_TABLE_PATH,
)


def main():
    if not os.path.exists(Q_TABLE_PATH):
        print(f"ERROR: No Q-table found at '{Q_TABLE_PATH}'.")
        print("       Run  python src/dynamic_obstacle_v_cnst.py  first.")
        sys.exit(1)

    print(f"Loading Q-table from {Q_TABLE_PATH} ...")
    q_table = load_q_table()
    print(f"  {len(q_table)} states loaded.\n")

    # Visual evaluation — 10 rendered episodes (Pygame window)
    print("--- Visual evaluation (10 episodes) ---")
    evaluate(q_table, episodes=10)

    # Decision accuracy vs rule-based baseline (headless)
    print("\nComputing decision accuracy (50 headless episodes)...")
    accuracy = compute_decision_accuracy(q_table, episodes=50)
    print(f"Decision accuracy (vs rule-based): {accuracy:.1%}")


if __name__ == "__main__":
    main()
