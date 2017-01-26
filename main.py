from tools.methods import DeepLearning

def test():
    DIRECTORY = '../images/training_2014_09_20'
    dl = DeepLearning('conv3')
    dl.train(DIRECTORY, 50)


if __name__ == '__main__':
    test()