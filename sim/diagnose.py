import numpy as np
import os
import sys

def diagnose():
    qtable_path = os.path.join(os.path.dirname(__file__), 'data', 'q_table.npy')
    if not os.path.exists(qtable_path):
        print("Q-table not found.")
        return
        
    qtable = np.load(qtable_path)
    
    # Visited states: those where not all Q-values are 0
    visited_mask = ~np.all(qtable == 0, axis=1)
    visited_states = np.where(visited_mask)[0]
    n_visited = len(visited_states)
    
    if n_visited == 0:
        print("No states visited.")
        return
        
    q_range = np.max(qtable) - np.min(qtable)
    threshold = 0.01 * q_range
    
    small_gap_count = 0
    for s in visited_states:
        q_vals = np.sort(qtable[s])
        gap = q_vals[-1] - q_vals[-2]
        if gap < threshold:
            small_gap_count += 1
            
    pct = (small_gap_count / n_visited) * 100
    print(f"Total states: {len(qtable)}")
    print(f"Visited states: {n_visited}")
    print(f"Q-value range: {q_range:.4f} (1% = {threshold:.4f})")
    print(f"States with gap < 1%: {small_gap_count} ({pct:.2f}%)")

if __name__ == '__main__':
    diagnose()
