import sys
from src.preprocessing import *
from src.test import *
from src.train import *


def main():

    if sys.argv[1] == 'pre':
        preprocess()
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'test':
        test()


if __name__ == '__main__':
    main()


