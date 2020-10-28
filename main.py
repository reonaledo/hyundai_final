import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

from src.preprocessing import *
from src.test import *
from src.train import *


def main():
    if len(sys.argv) < 2:
        print('수행할 작업을 선택해주세요. --preprocess / --train / --test')
    if sys.argv[1] == '--preprocess':
        print('전처리 시작')
        preprocess()
    if sys.argv[1] == '--train':
        print('모델 학습 시작')
        train()
    if sys.argv[1] == '--test':
        print('모델 테스트 시작')
        test()

if __name__ == '__main__':
    main()


