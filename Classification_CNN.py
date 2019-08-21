import os
import time
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Shufflenet_V2 import ShufflenetV2
from TestClassificationDataset import TestClassificationDataset
from TestVerificationDataset import TestVerificationDataset


class Classification_CNN:
    def __init__(self):
        self.batch_size = 64
        self.num_workers = 8
        self.train_data_params = {'batch_size': self.batch_size,
                                  'shuffle': True,
                                  'num_workers': self.num_workers,
                                  'pin_memory': True}
        self.val_data_params = {'batch_size': self.batch_size,
                                'shuffle': False,
                                'num_workers': self.num_workers,
                                'pin_memory': True}
        self.test_data_params = {'batch_size': self.batch_size,
                                 'shuffle': False,
                                 'num_workers': self.num_workers,
                                 'pin_memory': True}
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomAffine(0, (0.1, 0.1)),
            torchvision.transforms.ToTensor()
        ])

        print('Creating the training dataset.')
        training_gen_start_time = time.time()
        self.training_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/hw2/hw2p2_check/train_data/medium/',
                                                            transform=transforms)
        print('Creating the training dataset took ' + repr(time.time() - training_gen_start_time) + ' seconds.')

        print('Creating the validation dataset.')
        self.validation_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/hw2/hw2p2_check/validation_classification/medium/',
                                                              transform=torchvision.transforms.ToTensor())
        num_classes = len(self.training_dataset.classes)
        print('Num classes is ' + repr(num_classes) + '.')
        print('Num training batches per epoch is ' + repr(len(self.training_dataset.samples) / self.batch_size) + '.')

        self.net = ShufflenetV2(num_classes=num_classes, input_size=32, width_mult=2.0)

        self.criterion = nn.CrossEntropyLoss()
        self.net.apply(init_weights)

        self.cos = nn.CosineSimilarity()

    # def verification_train(self, epochs, gpu, val_trials, test_trials, lr=1e-4):
    #     device = torch.device('cuda' if gpu else 'cpu')
    #     self.net = self.net.to(device)

    def train(self, epochs, gpu, lr=0.001, weight_decay=1e-5, momentum=0):
        device = torch.device('cuda' if gpu else 'cpu')
        self.net = self.net.to(device)

        training_generator = DataLoader(self.training_dataset, **self.train_data_params)
        validation_generator = DataLoader(self.validation_dataset, **self.val_data_params)

        optimizer = torch.optim.Adam(self.net.parameters(), lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        epochs_between_validation = 1

        print('Beginning training.')
        for epoch in range(epochs):
            print('Device is ' + repr(device))
            start = time.time()
            count = 0
            cumulative_train_accuracy = 0.0
            cumulative_train_loss = 0.0
            for batch, labels in training_generator:
                self.net.train()
                if (count % 2000 == 0 and count > 0):
                    print("So far, training on {:} batches has taken {:.2f} minutes. Average training accuracy is {:.4f}"
                          .format(count, (time.time() - start) / 60, cumulative_train_accuracy/count))
                batch, labels = batch.to(device), labels.to(device)

                output = self.net(batch, False)
                loss = self.criterion(output, labels)
                cumulative_train_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
                optimizer.step()
                optimizer.zero_grad()
                count += 1

                output = output.cpu().detach()
                prediction = torch.argmax(output, dim=1)
                accuracy = (prediction.numpy() == labels.cpu().numpy()).mean()
                cumulative_train_accuracy += accuracy

            print("After epoch ", repr(epoch))
            print("Training loss :", repr(cumulative_train_loss / count))
            print("Training accuracy :", repr(cumulative_train_accuracy / count))

            if epoch % epochs_between_validation == 0:
                self.net.eval()
                cumulative_val_accuracy = 0.0
                cumulative_val_loss = 0.0
                val_count = 0
                with torch.set_grad_enabled(False):
                    for val_batch, val_labels in validation_generator:
                        val_batch, val_labels = val_batch.to(device), val_labels.to(device)

                        val_output = self.net(val_batch, False)
                        val_loss = self.criterion(val_output, val_labels)
                        cumulative_val_loss += val_loss.item()

                        val_output = val_output.cpu().detach()
                        val_prediction = torch.argmax(val_output, dim=1)
                        val_accuracy = (val_prediction.numpy() == val_labels.cpu().numpy()).mean()
                        cumulative_val_accuracy += val_accuracy
                        val_count += 1

                print("Validation loss: ", repr(cumulative_val_loss / val_count))
                print("Validation accuracy: ", repr(cumulative_val_accuracy / val_count))
                scheduler.step(cumulative_val_loss / val_count)

            stop = time.time()
            print("This epoch took " + repr((stop - start) / 60) + " minutes.")
            backup_file = 'model_' + repr(epoch) + 'valAcc_' + repr(cumulative_val_accuracy/val_count) + '.pt'
            torch.save(self.net.state_dict(), backup_file)
        self.net = self.net.cpu()
        print("Finished training.")

    def test(self, gpu, is_verifying, root, trials=None):
        device = torch.device('cuda' if gpu else 'cpu')
        self.net = self.net.to(device)
        self.net.eval()

        print('Creating the testing generator.')
        if not is_verifying:
            test_dataset = TestClassificationDataset(root)
        else:
            test_dataset = TestVerificationDataset(root, trials)
        test_generator = DataLoader(test_dataset, **self.test_data_params)
        print('Num testing batches is ' + repr(test_dataset.__len__() / self.batch_size))

        print('Beginning testing.')
        count = 0
        out_line = 0
        start = time.time()
        if not is_verifying:
            out_file = open("classification_output.csv", "w")
            out_file.write("id,label\n")
            out_file.close()
        else:
            out_file = open("verification_output.csv", "w")
            out_file.write("trial,score\n")
            out_file.close()
            
        for test_batch in test_generator:
            if (count % 1000 == 0):
                print("So far, testing on {:} batches has taken {:.2f} minutes.".format(count, (time.time()-start)/60))

            if not is_verifying:
                test_batch = test_batch.to(device)
                test_output = self.net(test_batch, is_verifying)
                test_output = test_output.cpu().detach()
                test_predictions = torch.argmax(test_output, dim=1)
                out_file = open("classification_output.csv", "a+")
                for test_prediction in test_predictions:
                    out = repr(out_line) + "," + self.validation_dataset.classes[test_prediction.item()] + "\n"
                    out_file.write(out)
                    out_line += 1
                out_file.close()
            else:
                a = test_batch[0]
                b = test_batch[1]
                a = a.to(device)
                b = b.to(device)
                a_out = self.net(a, is_verifying)
                b_out = self.net(b, is_verifying)
                a_out = a_out.cpu()
                b_out = b_out.cpu()
                distances = self.cos(a_out, b_out)
                distances = distances.cpu().detach()
                out_file = open("verification_output.csv", "a+")
                for distance in distances:
                    trial = test_dataset.get_trial(out_line)
                    trial = repr(trial[0]) + ".jpg " + repr(trial[1]) + ".jpg"
                    out = trial + "," + repr(distance.item()) + "\n"
                    out_file.write(out)
                    out_line += 1
                out_file.close()

            count += 1

        print('Finished testing.')
        self.net = self.net.cpu()

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
