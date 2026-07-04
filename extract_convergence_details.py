# extract_convergence_details.py
import json
import pandas as pd
import numpy as np

def extract_details():
    with open("results/speed_benchmark_data.json", 'r') as f:
        results = json.load(f)
        
    configs = ["QUEST_4x4", "RMax_4x4", "QUEST_8x8", "RMax_8x8"]
    
    for name in configs:
        print(f"\n=== Convergence details for {name} ===")
        threshold = 0.70 if "4x4" in name else 0.85
        
        seeds_conv_ep = []
        seeds_conv_time = []
        
        for idx, seed_data in enumerate(results[name]["seeds"]):
            rewards = seed_data["rewards"]
            times = seed_data["cumulative_times"]
            
            rolling_avg = pd.Series(rewards).rolling(window=100).mean().fillna(0).values
            cross_indices = np.where(rolling_avg >= threshold)[0]
            
            if len(cross_indices) > 0:
                conv_ep = cross_indices[0] + 1
                conv_time = times[cross_indices[0]]
            else:
                conv_ep = len(rewards)
                conv_time = times[-1]
                
            seeds_conv_ep.append(conv_ep)
            seeds_conv_time.append(conv_time)
            print(f"  Seed {idx}: Episode {conv_ep} | Time {conv_time:.2f}s")
            
        print(f"  Mean Convergence Episode: {np.mean(seeds_conv_ep):.1f}")
        print(f"  Mean Convergence Time: {np.mean(seeds_conv_time):.2f}s")

if __name__ == "__main__":
    extract_details()
