# A script that given a directory, will launch all the experiments in that directory

import argparse
import os
import sys
from pathlib import Path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', '-e', type=str, required=True)
    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    assert exp_path.exists()
    assert exp_path.is_dir()

    exp_parts = exp_path.parts
    config_part = exp_parts.index('config')
    exp_name = '/'.join(exp_parts[config_part + 1:])
    for file in os.listdir(exp_path):
        if file.endswith('.yaml') and not file.startswith('_'):
            os.system(f'sbatch ./crc_scripts/launch_v100.sh +{exp_name}={file[:-5]}')

if __name__ == '__main__':
    main()

