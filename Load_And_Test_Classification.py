import sys
import torch
from Classification_CNN import Classification_CNN


def main():
    path = sys.argv[1]
    classification_cnn = Classification_CNN()
    classification_cnn.net.load_state_dict(torch.load(path))
    test_root = '/home/ubuntu/hw2/hw2p2_check/test_classification/medium/'
    classification_cnn.test(True, False, test_root)

if __name__ == '__main__':
    main()