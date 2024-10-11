import os
import json
import numpy as np
import argparse

from util import update_index

parser = argparse.ArgumentParser(description='Reproducibility setup')
parser.add_argument('--test_dir', default=None, type=str,
                    help='Path to audio directory')
parser.add_argument('--ir_dir', default=None, type=str,
                    help='Path to impulse response directory')
parser.add_argument('--noise_dir', default=None, type=str,
                    help='Path to noise directory')
parser.add_argument('--eval_type', default='fma_medium', type=str,
                    help='Evaluation type (medium or large)')
                    

def main():
    args = parser.parse_args()

    # Check data files and directories are in order
    if not os.path.exists('data'):
        os.mkdir('data')
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if not os.path.exists('runs'):
        os.mkdir('runs')

    # Update index files with new parent directory
    _ = update_index(args.test_dir, f'data/{args.eval_type}.json')
    _ = update_index(args.ir_dir, f'data/ir.json')
    _ = update_index(args.noise_dir, f'data/noise.json')


if __name__ == '__main__':
    main()