# A script that given a directory, will launch all the experiments in that directory

import argparse
import os
import sys
from pathlib import Path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', '-e', type=str, required=True)
    parser.add_argument('--seed', '-s', type=int, default=-1)
    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    assert exp_path.exists()
    assert exp_path.is_dir()

    exp_parts = exp_path.parts
    config_part = exp_parts.index('config')
    exp_name = '/'.join(exp_parts[config_part + 1:])
    for file in os.listdir(exp_path):
        if file.endswith('.yaml') and not file.startswith('_'):
            launch_cmd = f'sbatch ./launch.sh +{exp_name}={file[:-5]}'
            if args.seed != -1:
                if exp_name[-3:-1] == '_s': # seed suffix
                    exp_name = exp_name[:-3]
                launch_cmd += f' seed={args.seed} experiment_set={exp_name[len("exp/"):]}_s{args.seed}'
            os.system(launch_cmd)

if __name__ == '__main__':
    main()

