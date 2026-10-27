import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from offload_env import state_to_index, N_BATTERY_BINS, N_NETWORKS

def print_heatmaps():
    qtable_path = os.path.join(os.path.dirname(__file__), 'data', 'q_table.npy')
    qtable = np.load(qtable_path)
    
    panels = [
        (0, 0, "Light Task, Empty Queue"),
        (2, 0, "Heavy Task, Empty Queue"),
        (2, 4, "Heavy Task, Full Queue"),
        (1, 2, "Medium Task, Med Queue")
    ]
    
    net_labels = ["Poor", "Medium", "Good"]
    bat_labels = ["<20%", "20-40%", "40-60%", "60-80%", ">80%"]
    
    for task, q, title in panels:
        print(f"\n--- {title} ---")
        print("          Poor  Medium  Good")
        for bat in range(N_BATTERY_BINS):
            # Print inverted Y-axis as in the heatmap (top to bottom is >80% to <20%, wait plot_heatmaps does ax.invert_yaxis() on 0=0 to 4=4?
            # sns.heatmap without invert_yaxis has 0 at top. invert_yaxis puts 0 at bottom.
            # Usually index 0 (<20%) is at the bottom. 
            row_actions = []
            for net in range(N_NETWORKS):
                idx = state_to_index(bat, net, task, q, 0)
                best_action = np.argmax(qtable[idx])
                row_actions.append(best_action)
            
            bat_lbl = bat_labels[bat].ljust(8)
            print(f"{bat_lbl}:  {row_actions[0]}       {row_actions[1]}       {row_actions[2]}")

if __name__ == '__main__':
    print_heatmaps()
