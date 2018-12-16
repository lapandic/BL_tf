#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime
from cnn_training_models import *

def load_data(file_path,steps_ahead):
    '''
    Loads images and velocities from hdf files and checks for potential mismatch in the number of images and velocities

    :param file_path: path to the hdf file from which it will extract the data
    :return: velocities, images as numpy arrays
    '''

    # read dataframes
    df_data = pd.read_hdf(file_path, key='data')
    df_img = pd.read_hdf(file_path, key='images', encoding='utf-8')

    # extract omega velocities from dataset
    velocities = np.transpose([df_data['vel_omega_'+str(1)].values])
    for i in range(1,steps_ahead):
        velocities = np.append(velocities, np.transpose([df_data['vel_omega_'+str(i+1)].values]),axis=1)

    # extract images from dataset
    images = df_img.values

    print('The dataset is loaded: {} images and {} omega velocities.'.format(images.shape, velocities.shape))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    return velocities, images


def form_model_name(batch_size, lr, optimizer, epochs,history,arch_num,depth,batch_normalization,steps_ahead):
    '''
    Creates name of model as a string, based on the defined hyperparameters used in training

    :param batch_size: batch size
    :param lr: learning rate
    :param optimizer: optimizer (e.g. GDS, Adam )
    :param epochs: number of epochs
    :param history: number of HISTORY
    :return: name of model as a string
    '''

    #return "batch={},lr={},optimizer={},epochs={},HISTORY={}".format(batch_size, lr, optimizer, epochs,history)
    return "datetime={},bs={},fs={},arch_num={},lr={},opt={}".format(datetime.datetime.now().strftime("%y%m%d%H%M"),history,steps_ahead,arch_num,lr,optimizer)

class CNN_training:

    def __init__(self, batch, epochs, learning_rate, optimizer,history,arch_num,use_batch_normalization,steps_ahead):

        self.batch_size = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.history = history
        self.arch_num = arch_num
        self.use_batch_normalization = bool(use_batch_normalization)
        self.img_row_width = 48*96*3
        self.loss = None
        self.loss_train = None
        self.loss_test = None
        self.steps_ahead = steps_ahead


        self.model = get_model(history,steps_ahead,arch_num)

    def backpropagation(self):
        '''
        Executes backpropagation during training based on the defined optimizer,learning rate and loss function

        '''

        # define the optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_train)
            elif self.optimizer == "GDS":
                return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss_train)

    def loss_function(self,training=False):
        '''
        Calculates the loss during training using the predicted and true values(in this case velocities)

        '''

        # define loss function and encapsulate its scope
        with tf.name_scope("Loss"):
            if self.use_batch_normalization:
                if training:
                    return tf.reduce_mean(tf.square(self.vel_pred_train - self.vel_true))
                else:
                    return tf.reduce_mean(tf.square(self.vel_pred_test - self.vel_true))
            else:
                return tf.reduce_mean( tf.square(self.vel_pred - self.vel_true) )

    def epoch_iteration(self, data_size, x_data, y_data, mode):
        '''
        For each epoch extract batches and execute train or test step depending on the inserted mode

        :param data_size: number of velocities and images
        :param x_data: images
        :param y_data: velocities
        :param mode: 'train' or 'test' in order to define if backpropagation is executed as well or not
        :return: sum of loss at each epoch
        '''

        pred_loss = 0
        i = 0
        while i <= data_size - 1:

            # extract batch
            if i + self.batch_size <= data_size - 1:
                train_x = x_data[i: i + self.batch_size]
                train_y = y_data[i: i + self.batch_size]
            else:
                train_x = x_data[i:]
                train_y = y_data[i:]

            if mode == 'train':
                # train using the batch and calculate the loss
                _, c = self.sess.run([self.opt, self.loss_train], feed_dict={self.x: train_x, self.vel_true: train_y})

            elif mode == 'test':
                # train using the batch and calculate the loss
                c = self.sess.run(self.loss_test, feed_dict={self.x: train_x, self.vel_true: train_y})

            pred_loss += c
            i += self.batch_size

        return pred_loss

    def training(self, model_name, train_velocities, train_images, test_velocities, test_images):

        # define paths to save the TensorFlow logs
        model_path = os.path.join(os.getcwd(), 'tensorflow_logs', model_name)
        logs_train_path = os.path.join(model_path, 'train')
        logs_test_path = os.path.join(model_path, 'test')
        graph_path = os.path.join(model_path, 'graph')


        # manual scalar summaries for loss tracking
        man_loss_summary = tf.Summary()
        man_loss_summary.value.add(tag='Loss', simple_value=None)

        # define placeholder variable for input images (each images is a row vector [1, 4608 = 48x96x1])
        self.x = tf.placeholder(tf.float16, shape=[None, 48 * 96 * 3*self.history], name='x')
        # self.x = []
        #for i in range(0,self.history):
        # self.x.append(tf.placeholder(tf.float16, shape=[None, 48 * 96 * 3], name="x"+str(i)))



        # define placeholder for the true omega velocities
        # [None: tensor may hold arbitrary num of velocities, number of omega predictions for each image]
        self.vel_true = tf.placeholder(tf.float16, shape=[None, self.steps_ahead], name="vel_true")
        if self.use_batch_normalization:
            self.vel_pred_train = self.model(self.x, training=True)
            self.vel_pred_test = self.model(self.x, training=False)
            self.loss_train = self.loss_function(training=True)
            self.loss_test = self.loss_function(training=False)
        else:
            if self.arch_num > 9:
                self.vel_pred = self.model(self.x,training=False)
            else:
                self.vel_pred = self.model(self.x)
            self.loss_train = self.loss_function()
            self.loss_test = self.loss_function()

        self.opt = self.backpropagation()

        # initialize variables
        init = tf.global_variables_initializer()

        # Operation to save and restore all variables
        saver = tf.train.Saver()

        with tf.Session() as self.sess:

            # run initializer
            self.sess.run(init)

            # operation to write logs for Tensorboard
            tf_graph = self.sess.graph
            test_writer = tf.summary.FileWriter(logs_test_path, graph=tf.get_default_graph() )
            test_writer.add_graph(tf_graph)

            train_writer = tf.summary.FileWriter(logs_train_path, graph=tf.get_default_graph() )
            train_writer.add_graph(tf_graph)

            # IMPORTANT: this is a crucial part for compiling TensorFlow graph to a Movidius one later in the pipeline.
            # The important file to create is the 'graph.pb' which will be used to freeze the TensorFlow graph.
            # The 'graph.pbtxt' file is just the same graph in txt format in case you want to check the format of the
            # saved information.
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pbtxt', as_text= True)
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pb', as_text= False)

            for epoch in range(self.epochs):

                # run train cycle
                avg_train_loss = self.epoch_iteration(train_velocities.shape[0], train_images, train_velocities, 'train')

                # save the training loss using the manual summaries
                man_loss_summary.value[0].simple_value = avg_train_loss
                train_writer.add_summary(man_loss_summary, epoch)

                # run test cycle
                avg_test_loss = self.epoch_iteration(test_velocities.shape[0], test_images, test_velocities, 'test')

                # save the test errors using the manual summaries
                man_loss_summary.value[0].simple_value = avg_test_loss
                test_writer.add_summary(man_loss_summary, epoch)

                # print train and test loss to monitor progress during training every 50 epochs
                #if epoch % 50 == 0:
                print("Epoch: {:04d} , train_loss = {:.6f} , test_loss = {:.6f}".format(epoch+1, avg_train_loss, avg_test_loss))

                # save weights every 100 epochs
                if epoch % 100 == 0:
                    saver.save(self.sess, logs_train_path, epoch)

            model_vars = tf.trainable_variables()
            tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        # close summary writer
        train_writer.close()
        test_writer.close()

