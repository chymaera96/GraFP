import argparse
import yaml
from util import load_config

def main(args):

    for cfg_path in ['config/grafp.yaml', 'config/ast.yaml']:
        config = load_config(cfg_path)
        if args.train_dir is not None:
            config['train_dir'] = args.train_dir
        if args.val_dir is not None:
            config['val_dir'] = args.val_dir
        config['noise_dir'] = args.noise_dir
        config['ir_dir'] = args.ir_dir

        with open(cfg_path, 'w') as fp:
            yaml.dump(config, fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=False, default=None)
    parser.add_argument('--val_dir', required=False, default=None)
    parser.add_argument('--noise_dir', required=True)
    parser.add_argument('--ir_dir', required=True)

    args = parser.parse_args()

    main(args)