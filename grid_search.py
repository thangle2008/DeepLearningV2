import argparse

from hyperopt import hp
from tools.methods import DeepLearning

def run(args):
    space = {
        'batch_size': 32,
        'num_epochs': 100,
        'optimizer': hp.choice('optimizer', [
            {
                'type': 'adam',
                'params': {
                'lr': 0.0017249684527549864,
                'epsilon': 1e-03
                }
            },
            {
                'type': 'adagrad',
                'params': {
                'lr': 0.008358387907938093,
                'epsilon': 1e-03
                }
            },
            {
                'type': 'adadelta',
                'params': {
                'lr': 0.07857133605571984,
                'epsilon': 1e-03
                }
            }
        ])
        # 'optimizer': {
        #     'type': 'adam',
        #     'params': {
        #         'lr': hp.uniform('lr', 0.0005, 0.005),
        #         'epsilon': 1e-03
        #     }
        # }
    }
    dl = DeepLearning('conv3')
    dl.grid_search(args.dir, space, max_evals=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', action='store', dest='dir',
                        help='data folder', required=True)

    args = parser.parse_args()

    run(args)