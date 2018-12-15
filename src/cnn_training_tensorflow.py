#!/usr/bin/env python

import time
import os
import argparse
from cnn_training_functions import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(arch_num,history,depth,lr,epochs_n,use_batch_normalization,steps_ahead):

    # define path for training dataset
    file_path_train = os.path.join(os.getcwd(), 'data-'+str(history), 'train', 'train_set.h5')
    file_path_test = os.path.join(os.getcwd(), 'data-'+str(history), 'test', 'test_set.h5')

    # define batch_size (e.g 50, 100)
    batch_size = 100

    # define which optimizer you want to use (e.g "Adam", "GDS"). For "Adam" and "GDS" this script will take care the rest.
    # ATTENTION !! If you want to choose a different optimizer from these two, you will have to add it in the training functions.
    optimizer = "GDS"

    # define learning rate (e.g 1E-3, 1E-4, 1E-5):
    learning_rate = lr

    # define total epochs (e.g 1000, 5000, 10000)
    epochs = epochs_n

    # read train data
    print('Reading train dataset')
    train_velocities, train_images = load_data(file_path_train)
    train_velocities = np.reshape(train_velocities, (-1, steps_ahead))

    # read test data
    print('Reading test dataset')
    test_velocities, test_images = load_data(file_path_test)
    test_velocities = np.reshape(test_velocities, (-1, steps_ahead))

    # construct model name based on the hyper parameters
    model_name = form_model_name(batch_size, learning_rate, optimizer, epochs,history,arch_num,depth,use_batch_normalization,steps_ahead)

    print('Starting training for {} model.'.format(model_name))

    # keep track of training time
    start_time = time.time()

    # train model
    cnn_train = CNN_training(batch_size, epochs, learning_rate, optimizer, history,arch_num,use_batch_normalization,steps_ahead)
    cnn_train.training(model_name, train_velocities, train_images, test_velocities, test_images)

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60

    print('Finished training of {} in {} minutes.'.format(model_name, training_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs','--backsteps',default=1, help='Number of steps in past', type=int)
    parser.add_argument('-fs', '--steps_ahead', default=1, help='Number of steps in future', type=int)
    parser.add_argument('-a', '--arch_num', default=0, help='Unique id number of architecture', type=int)
    parser.add_argument('-d', '--depth', default=1, help='Depth', type=int)
    parser.add_argument('-e', '--epochs', default=1000, help='Number of epochs', type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, help='Learning rate', type=float)
    parser.add_argument('-gpu', '--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument('-bn', '--batch_normalization', default=0, help='Use batch normalization', type=int)
    args = vars(parser.parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
    main(args['arch_num'],args['backsteps'],args['depth'],args['learning_rate'],args['epochs'],args['batch_normalization'],args['steps_ahead'])
