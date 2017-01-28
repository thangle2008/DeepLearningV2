import json
import argparse

from tools.methods import DeepLearning

def run(args):

    # load config file
    f = open(args.config, 'r')
    config = json.load(f)

    # load model and train network
    dl = DeepLearning(config['model'])

    dl.train(args.dir, config['num_epoches'], batch_size=config['batch_size'],
            optimizer=config['optimizer'], options=config['options'], seed=42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store', dest='config',
                        help='config file for training', required=True)
    parser.add_argument('-f', '--folder', action='store', dest='dir',
                        help='data folder', required=True)
    args = parser.parse_args()

    run(args)