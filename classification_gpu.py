import sys

from Classification_CNN import Classification_CNN


def main():
    classification_cnn = Classification_CNN()
    print("Initialized classification CNN.")

    epochs = 40
    gpu = True
    lr = .001
    wd = 1e-5
    if len(sys.argv) == 1:
        is_verifying = False
        train_root = 'hw2p2_train_data/medium/'
        classification_cnn.train(epochs, gpu, lr, wd, train_root)
        test_root = 'hw2p2_check/test_classification/medium/'
        classification_cnn.test(gpu, is_verifying, test_root)
    else:
        # val_trials = 'hw2p2_check/val_trials_verification_student.txt'
        test_trials = 'hw2p2_check/test_trials_verification_student.txt'
        root = 'hw2p2_check/test_verification/'
        is_verifying = True
        # ver_epochs = 4
        # classification_cnn.verification_train(ver_epochs, gpu, val_trials, test_trials)
        classification_cnn.test(gpu, is_verifying, root, test_trials)

if __name__ == '__main__':
    main()
