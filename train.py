import json
import argparse

from hyperopt import hp
from tools.methods import DeepLearning

def run(args):

    # load config file
    f = open(args.config, 'r')
    config = json.load(f)

    optimizer = config['optimizer']

    # load model and train network
    dl = DeepLearning(config['model'])
    dl.train(args.dir, optimizer['type'], config['num_epochs'], config['batch_size'],
                target_size=tuple(config['target_size']), crop_dim=config['crop_dim'],
                **optimizer['params'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store', dest='config',
                        help='config file for training', required=True)
    parser.add_argument('-f', '--folder', action='store', dest='dir',
                        help='data folder', required=True)
    args = parser.parse_args()

    run(args)