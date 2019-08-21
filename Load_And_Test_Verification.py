import sys
import torch
from Classification_CNN import Classification_CNN


def main():
    path = sys.argv[1]
    classification_cnn = Classification_CNN()
    classification_cnn.net.load_state_dict(torch.load(path))
    test_trials = '/home/ubuntu/hw2/hw2p2_check/test_trials_verification_student.txt'
    test_root = '/home/ubuntu/hw2/hw2p2_check/test_verification/'
    classification_cnn.test(True, True, test_root, test_trials)


if __name__ == '__main__':
    main()