import argparse

from hyperopt import hp
from tools.methods import DeepLearning

def run(args):
    space = {
        'batch_size': 32,
        'num_epochs': 50,
        'optimizer': 'adagrad',
        'lr': hp.uniform('lr', 0.003, 0.03),
        'epsilon': 1e-03
    }
    dl = DeepLearning('conv3')
    dl.grid_search(args.dir, space, 10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', action='store', dest='dir',
                        help='data folder', required=True)

    args = parser.parse_args()

    run(args)