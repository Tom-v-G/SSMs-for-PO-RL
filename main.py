import argparse
import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from utils import load_yaml_config
from algorithms.PPO_V2_done_signal import train

import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SSM Suite")
    parser.add_argument(
        "-p", "-cfg", "-path" , type=str, default="", help="Path to yaml config files"
    )
    args = parser.parse_args()

    configs_dir = args.p

    for subdir, dirs, files in os.walk(args.p):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(f".yaml"):
                cfg = load_yaml_config(filepath)
                print(cfg)
                print(filepath)
                for i in range(cfg.num_reps):
                    cfg = load_yaml_config(filepath) # to reset seed
                    train(cfg)
        
