import sys

from Classification_CNN import Classification_CNN


def main():
    classification_cnn = Classification_CNN()
    print("Initialized classification CNN.")
    epochs = 0
    gpu = False
    lr = .001
    wd = 1e-5
    if len(sys.argv) == 1:
        is_verifying = False
        classification_cnn.train(epochs, gpu, lr, wd)
        root = 'hw2p2_check/test_classification/medium/'
        classification_cnn.test(gpu, is_verifying, root)
    else:
        trials = 'hw2p2_check/test_trials_verification_student.txt'
        root = 'hw2p2_check/test_verification/'
        is_verifying = True
        classification_cnn.test(gpu, is_verifying, root, trials)

if __name__ == '__main__':
    main()