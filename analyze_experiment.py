import numpy as np
import pandas as pd
import argparse
from config import Config, load_config

def parse():
    parser = argparse.ArgumentParser(description="Simulating Jolly-Seber")
    parser.add_argument("--sim_data_dir", default="sim_data")
    parser.add_argument("--experiment_name", default="tmp")
    parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()

def main():

    args = parse()
    cfg = load_config(args.config_path, "config/default.yaml")

    return None

if __name__ == '__main__':
    main()