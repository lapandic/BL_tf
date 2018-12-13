#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime

def load_data(file_path):
    '''
    Loads images and velocities from hdf files and checks for potential mismatch in the number of images and velocities

    :param file_path: path to the hdf file from which it will extract the data
    :return: velocities, images as numpy arrays
    '''

    # read dataframes
    df_data = pd.read_hdf(file_path, key='data')
    df_img = pd.read_hdf(file_path, key='images', encoding='utf-8')

    # extract omega velocities from dataset
    velocities = df_data['vel_omega'].values
    velocities = np.reshape(velocities, (-1, 1))

    # extract images from dataset
    images = df_img.values

    print('The dataset is loaded: {} images and {} omega velocities.'.format(images.shape[0], velocities.shape[0]))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    return velocities, images


def form_model_name(batch_size, lr, optimizer, epochs,history,arch_num,depth,batch_normalization):
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
    return "datetime={},arch_num={},history={},depth={},lr={},opt={},bn={}".format(datetime.datetime.now().strftime("%y%m%d%H%M"),arch_num,history,depth,lr,optimizer,batch_normalization)

class CNN_training:

    def __init__(self, batch, epochs, learning_rate, optimizer,history,arch_num,use_batch_normalization):

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


        self.models = [self.model1_h1_d1_n1, self.model2_h1_d2_n1, self.model2_h1_d2_n2, self.model2_h2_d1_n3,
                       self.model3_h1_d3_n1, self.model3_h1_d3_n2,
                       self.model3_h1_d3_n3, self.model3_h3_d1_n4, self.model3_h2_d2_n5, self.model3_h2_d2_n6,
                       self.model3_h2_d2_n7, self.model3_h2_d2_n8, self.model3_h2_d2_n9]

        self.models_h1 = [self.model1_h1_d1_n1, self.model2_h1_d2_n1, self.model3_h1_d3_n1, self.model3_h1_d3_n2,
                       self.model3_h1_d3_n3, self.model7_h1]

        self.models_h2 = [self.model2_h2_d1_n3, self.model3_h2_d2_n5, self.model3_h2_d2_n6,
                       self.model3_h2_d2_n7, self.model3_h2_d2_n8, self.model3_h2_d2_n9]

        self.models_h3 = [self.model3_h3_d1_n4]

        self.models_dronet = [self.model_dronet, self.model_dronet2, self.model_dronet_1rb]

        if history == 1 and arch_num < 6:
            self.model = self.models_h1[arch_num]
        elif history == 2 and arch_num < 6:
            self.model = self.models_h2[arch_num]
        elif history == 3:
            self.model = self.model3_h3_d1_n4
        elif history == 1 and arch_num > 9:
            self.model = self.models_dronet[arch_num-10]
        else:
            print("Requested model not implemented!")

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
        self.vel_true = tf.placeholder(tf.float16, shape=[None, 1], name="vel_true")
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

        # close summary writer
        train_writer.close()
        test_writer.close()


    def model1_h1_d1_n1(self, x):
        '''
        Define model of CNN under the TensorFlow scope "ConvNet".
        The scope is used for better organization and visualization in TensorBoard

        :return: output layer
        '''

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # define 1st convolutional layer
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            conv_flat = tf.layers.flatten(max_pool_1)

            # add 2nd fully connected layers to predict the driving commands
            fc_1 = tf.layers.dense(inputs=conv_flat, units=1, name="fc_layer_out")

            return fc_1

    def model2_h1_d2_n1(self, x):
        '''
        Define model of CNN under the TensorFlow scope "ConvNet".
        The scope is used for better organization and visualization in TensorBoard

        :return: output layer
        '''

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # define 1st convolutional layer
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # define 2nd convolutional layer
            hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2")

            max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            conv_flat = tf.layers.flatten(max_pool_2)

            # add 2nd fully connected layers to predict the driving commands
            fc_1 = tf.layers.dense(inputs=conv_flat, units=1, name="fc_layer_out")

            return fc_1

    def model2_h1_d2_n2(self, x):
        '''
        Define model of CNN under the TensorFlow scope "ConvNet".
        The scope is used for better organization and visualization in TensorBoard

        :return: output layer
        '''

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # define 1st convolutional layer
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            conv_flat = tf.layers.flatten(max_pool_1)

            fc_n = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_layer_1")


            # add 2nd fully connected layers to predict the driving commands
            fc_1 = tf.layers.dense(inputs=fc_n, units=1, name="fc_layer_out")

            return fc_1

    def model2_h2_d1_n3(self, x):
        '''
        Define model of CNN under the TensorFlow scope "ConvNet".
        The scope is used for better organization and visualization in TensorBoard

        :return: output layer
        '''

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48*self.history, 96, 3])

            x_array = tf.split(x_img, num_or_size_splits=self.history,axis=1)

            hl_conv_1 = []
            max_pool_1 = []
            conv_flat = []

            for i in range(len(x_array)):

                # define 1st convolutional layer
                hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                             activation=tf.nn.relu, name="conv_layer_1_"+str(i)))

                max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

                # flatten tensor to connect it with the fully connected layers
                conv_flat.append(tf.layers.flatten(max_pool_1[i]))


            commands_stack = tf.concat(conv_flat,axis=1)

            fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

            return fc

    def model3_h1_d3_n1(self, x):
        '''
        Define model of CNN under the TensorFlow scope "ConvNet".
        The scope is used for better organization and visualization in TensorBoard

        :return: output layer
        '''

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # define 1st convolutional layer
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # define 2nd convolutional layer
            hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2")

            max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            hl_conv_3 = tf.layers.conv2d(max_pool_2, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_3")

            max_pool_3 = tf.layers.max_pooling2d(hl_conv_3, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            conv_flat = tf.layers.flatten(max_pool_3)

            # add 2nd fully connected layers to predict the driving commands
            fc_1 = tf.layers.dense(inputs=conv_flat, units=1, name="fc_layer_out")

            return fc_1

    def model3_h1_d3_n2(self, x):
        #CURRENT ARCHITECTURE
        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # f block
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2")

            max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            conv_flat = tf.layers.flatten(max_pool_2)

            # FC_n
            fc_n_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

            # FC_1
            fc_1 = tf.layers.dense(inputs=fc_n_1, units=1, name="fc_layer_out")

            return fc_1

    def model3_h1_d3_n3(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # f block
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            conv_flat = tf.layers.flatten(max_pool_1)

            # FC_n
            fc_n_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

            # FC_n
            fc_n_2 = tf.layers.dense(inputs=fc_n_1, units=64, activation=tf.nn.relu, name="fc_n_layer_2")

            # FC_1
            fc_1 = tf.layers.dense(inputs=fc_n_2, units=1, name="fc_layer_out")

            return fc_1

    def model3_h3_d1_n4(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48 * self.history, 96, 3])

            x_array = tf.split(x_img, num_or_size_splits=self.history, axis=1)

            hl_conv_1 = []
            max_pool_1 = []
            conv_flat = []

            for i in range(len(x_array)):
                # f block
                hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                                  activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

                max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

                conv_flat.append(tf.layers.flatten(max_pool_1[i]))

            commands_stack = tf.concat(conv_flat,axis=1)

            # FC_1
            fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

            return fc

    def model3_h2_d2_n5(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48 * self.history, 96, 3])

            x_array = tf.split(x_img, num_or_size_splits=self.history, axis=1)

            # f block pipe 1
            hl_conv_1 = tf.layers.conv2d(x_array[0], kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            conv_flat_1 = tf.layers.flatten(max_pool_1)

            # FC_n
            fc_n_1 = tf.layers.dense(inputs=conv_flat_1, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

            # f block pipe2
            hl_conv_2 = tf.layers.conv2d(x_array[1], kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2_1")

            max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            conv_flat_2 = tf.layers.flatten(max_pool_2)

            commands_stack = tf.concat([fc_n_1,conv_flat_2],axis=1)

            # FC_1
            fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

            return fc

    def model3_h2_d2_n6(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48 * self.history, 96, 3])

            x_array = tf.split(x_img, num_or_size_splits=self.history, axis=1)

            # f block pipe 1
            hl_conv_1_1 = tf.layers.conv2d(x_array[0], kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1_1")

            max_pool_1_1 = tf.layers.max_pooling2d(hl_conv_1_1, pool_size=2, strides=2)

            # f block pipe 1
            hl_conv_1_2 = tf.layers.conv2d(max_pool_1_1, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1_2")

            max_pool_1_2 = tf.layers.max_pooling2d(hl_conv_1_2, pool_size=2, strides=2)

            conv_flat_1 = tf.layers.flatten(max_pool_1_2)

            # f block pipe2
            hl_conv_2_1 = tf.layers.conv2d(x_array[1], kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2_1")

            max_pool_2_1 = tf.layers.max_pooling2d(hl_conv_2_1, pool_size=2, strides=2)

            conv_flat_2 = tf.layers.flatten(max_pool_2_1)

            commands_stack = tf.concat([conv_flat_1,conv_flat_2],axis=1)

            # FC_1
            fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

            return fc

    def model3_h2_d2_n7(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48 * self.history, 96, 3])

            x_array = tf.split(x_img, num_or_size_splits=self.history, axis=1)

            # f block pipe 1
            hl_conv_1_1 = tf.layers.conv2d(x_array[0], kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1_1")

            max_pool_1_1 = tf.layers.max_pooling2d(hl_conv_1_1, pool_size=2, strides=2)

            conv_flat_1 = tf.layers.flatten(max_pool_1_1)

            # f block pipe2
            hl_conv_2_1 = tf.layers.conv2d(x_array[1], kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2_1")

            max_pool_2_1 = tf.layers.max_pooling2d(hl_conv_2_1, pool_size=2, strides=2)

            # f block pipe 1
            hl_conv_2_2 = tf.layers.conv2d(max_pool_2_1, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2_2")

            max_pool_2_2 = tf.layers.max_pooling2d(hl_conv_2_2, pool_size=2, strides=2)

            conv_flat_2 = tf.layers.flatten(max_pool_2_2)

            commands_stack = tf.concat([conv_flat_1,conv_flat_2],axis=1)

            # FC_1
            fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

            return fc

    def model3_h2_d2_n8(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48 * self.history, 96, 3])

            x_array = tf.split(x_img, num_or_size_splits=self.history, axis=1)

            # f block pipe 1
            hl_conv_1_1 = tf.layers.conv2d(x_array[0], kernel_size=5, filters=2, padding="valid",
                                           activation=tf.nn.relu, name="conv_layer_1_1")

            max_pool_1_1 = tf.layers.max_pooling2d(hl_conv_1_1, pool_size=2, strides=2)

            conv_flat_1 = tf.layers.flatten(max_pool_1_1)

            # f block pipe2
            hl_conv_2_1 = tf.layers.conv2d(x_array[1], kernel_size=5, filters=2, padding="valid",
                                           activation=tf.nn.relu, name="conv_layer_2_1")

            max_pool_2_1 = tf.layers.max_pooling2d(hl_conv_2_1, pool_size=2, strides=2)

            conv_flat_2 = tf.layers.flatten(max_pool_2_1)

            # FC_n pipe2
            fc_n_1 = tf.layers.dense(inputs=conv_flat_2, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

            commands_stack = tf.concat([conv_flat_1, fc_n_1],axis=1)

            # FC_1
            fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

            return fc

    def model3_h2_d2_n9(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48 * self.history, 96, 3])

            x_array = tf.split(x_img, num_or_size_splits=self.history, axis=1)

            # f block pipe 1
            hl_conv_1_1 = tf.layers.conv2d(x_array[0], kernel_size=5, filters=2, padding="valid",
                                           activation=tf.nn.relu, name="conv_layer_1_1")

            max_pool_1_1 = tf.layers.max_pooling2d(hl_conv_1_1, pool_size=2, strides=2)

            conv_flat_1 = tf.layers.flatten(max_pool_1_1)

            # f block pipe2
            hl_conv_2_1 = tf.layers.conv2d(x_array[1], kernel_size=5, filters=2, padding="valid",
                                           activation=tf.nn.relu, name="conv_layer_2_1")

            max_pool_2_1 = tf.layers.max_pooling2d(hl_conv_2_1, pool_size=2, strides=2)

            conv_flat_2 = tf.layers.flatten(max_pool_2_1)

            commands_stack = tf.concat([conv_flat_1, conv_flat_2],axis=1)

            # FC_n
            fc_n_1 = tf.layers.dense(inputs=commands_stack, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

            # FC_1
            fc = tf.layers.dense(inputs=fc_n_1, units=1, name="fc_layer_out")

            return fc

    def model_dronet(self, x, training):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # first f block NOTE: using linear activation instead of ReLU. NOTE: number of filters!
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=32, padding="same",
                                         activation=None, name="conv_layer_1")

            # NOTE: using pool_size=2 instead of pool_size=3, reason input images in DroNet paper are 200x200x1, we use 48x96x3
            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)



            # Res block 1
            if training:
                rb_bn_1_1 = tf.layers.batch_normalization(max_pool_1, training=training)
            else:
                rb_bn_1_1 = max_pool_1
            rb_relu_1_1 = tf.nn.relu(rb_bn_1_1,name="rb_relu_1_1")
            rb_conv_1_1 = tf.layers.conv2d(rb_relu_1_1,kernel_size=3, filters=32, strides=2, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_1_1")

            if training:
                rb_bn_1_2 = tf.layers.batch_normalization(rb_conv_1_1, training=training)
            else:
                rb_bn_1_2 = rb_conv_1_1
            rb_relu_1_2 = tf.nn.relu(rb_bn_1_2,name="rb_relu_1_2")
            rb_conv_1_2 = tf.layers.conv2d(rb_relu_1_2,kernel_size=3, filters=32, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_1_2")

            rb_conv_1_3 = tf.layers.conv2d(max_pool_1, kernel_size=1, filters=32, strides=2, padding="same",
                                           name="rb_conv_1_3")

            rb_out_1 = tf.add(rb_conv_1_2,rb_conv_1_3,name="rb_out_1")

            # Res block 2
            if training:
                rb_bn_2_1 = tf.layers.batch_normalization(rb_out_1, training=training)
            else:
                rb_bn_2_1 = rb_out_1
            rb_relu_2_1 = tf.nn.relu(rb_bn_2_1,name="rb_relu_2_1")
            rb_conv_2_1 = tf.layers.conv2d(rb_relu_2_1,kernel_size=3, filters=64, strides=2, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_2_1")

            if training:
                rb_bn_2_2 = tf.layers.batch_normalization(rb_conv_2_1, training=training)
            else:
                rb_bn_2_2 = rb_conv_2_1
            rb_relu_2_2 = tf.nn.relu(rb_bn_2_2,name="rb_relu_2_2")
            rb_conv_2_2 = tf.layers.conv2d(rb_relu_2_2,kernel_size=3, filters=64, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_2_2")

            rb_conv_2_3 = tf.layers.conv2d(rb_out_1, kernel_size=1, filters=64, strides=2, padding="same",
                                           name="rb_conv_2_3")

            rb_out_2 = tf.add(rb_conv_2_2,rb_conv_2_3,name="rb_out_2")

            # Res block 3
            if training:
                rb_bn_3_1 = tf.layers.batch_normalization(rb_out_2, training=training)
            else:
                rb_bn_3_1 = rb_out_2
            rb_relu_3_1 = tf.nn.relu(rb_bn_3_1,name="rb_relu_2_1")
            rb_conv_3_1 = tf.layers.conv2d(rb_relu_3_1,kernel_size=3, filters=128, strides=2, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_3_1")

            if training:
                rb_bn_3_2 = tf.layers.batch_normalization(rb_conv_3_1, training=training)
            else:
                rb_bn_3_2 = rb_conv_3_1
            rb_relu_3_2 = tf.nn.relu(rb_bn_3_2,name="rb_relu_2_2")
            rb_conv_3_2 = tf.layers.conv2d(rb_relu_3_2,kernel_size=3, filters=128, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_3_2")

            rb_conv_3_3 = tf.layers.conv2d(rb_out_2, kernel_size=1, filters=128, strides=2, padding="same",
                                           name="rb_conv_3_3")

            rb_out_3 = tf.add(rb_conv_3_2,rb_conv_3_3,name="rb_out_3")

            # flatten
            flat =  tf.layers.flatten(rb_out_3)
            flat_relu = tf.nn.relu(flat, name="flat_relu_out")
            flat_dropout = tf.layers.dropout(flat_relu,name="flat_dropout_out")

            # FC_1 - predicting steering angle
            fc_1_steer = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

            # FC_1 - predicting steering angle
            # fc_1_collision = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

            return fc_1_steer

    def model_dronet2(self, x, training):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # first f block NOTE: using linear activation instead of ReLU. NOTE: number of filters!
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=32, padding="same",
                                         activation=None, name="conv_layer_1")

            # NOTE: using pool_size=2 instead of pool_size=3, reason input images in DroNet paper are 200x200x1, we use 48x96x3
            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)



            # Res block 1
            rb_bn_1_1 = tf.layers.batch_normalization(max_pool_1, training=training)
            rb_relu_1_1 = tf.nn.relu(rb_bn_1_1,name="rb_relu_1_1")
            rb_conv_1_1 = tf.layers.conv2d(rb_relu_1_1,kernel_size=3, filters=32, strides=2, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_1_1")

            rb_bn_1_2 = tf.layers.batch_normalization(rb_conv_1_1, training=training)
            rb_relu_1_2 = tf.nn.relu(rb_bn_1_2,name="rb_relu_1_2")
            rb_conv_1_2 = tf.layers.conv2d(rb_relu_1_2,kernel_size=3, filters=32, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_1_2")

            rb_conv_1_3 = tf.layers.conv2d(max_pool_1, kernel_size=1, filters=32, strides=2, padding="same",
                                           name="rb_conv_1_3")

            rb_out_1 = tf.add(rb_conv_1_2,rb_conv_1_3,name="rb_out_1")

            # Res block 2
            rb_bn_2_1 = tf.layers.batch_normalization(rb_out_1, training=training)
            rb_relu_2_1 = tf.nn.relu(rb_bn_2_1,name="rb_relu_2_1")
            rb_conv_2_1 = tf.layers.conv2d(rb_relu_2_1,kernel_size=3, filters=64, strides=2, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_2_1")

            rb_bn_2_2 = tf.layers.batch_normalization(rb_conv_2_1, training=training)
            rb_relu_2_2 = tf.nn.relu(rb_bn_2_2,name="rb_relu_2_2")
            rb_conv_2_2 = tf.layers.conv2d(rb_relu_2_2,kernel_size=3, filters=64, padding="same",
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),name="rb_conv_2_2")

            rb_conv_2_3 = tf.layers.conv2d(rb_out_1, kernel_size=1, filters=64, strides=2, padding="same",
                                           name="rb_conv_2_3")

            rb_out_2 = tf.add(rb_conv_2_2,rb_conv_2_3,name="rb_out_2")

            # flatten
            flat =  tf.layers.flatten(rb_out_2)
            flat_relu = tf.nn.relu(flat, name="flat_relu_out")
            flat_dropout = tf.layers.dropout(flat_relu,name="flat_dropout_out")

            # FC_1 - predicting steering angle
            fc_1_steer = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

            # FC_1 - predicting steering angle
            # fc_1_collision = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

            return fc_1_steer

    def model_dronet_1rb(self, x, training):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # first f block NOTE: using linear activation instead of ReLU. NOTE: number of filters!
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=32, padding="same",
                                         activation=None, name="conv_layer_1")

            # NOTE: using pool_size=2 instead of pool_size=3, reason input images in DroNet paper are 200x200x1, we use 48x96x3
            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # Res block 1
            if training:
                rb_bn_1_1 = tf.layers.batch_normalization(max_pool_1, training=training)
            else:
                rb_bn_1_1 = max_pool_1
            rb_relu_1_1 = tf.nn.relu(rb_bn_1_1, name="rb_relu_1_1")
            rb_conv_1_1 = tf.layers.conv2d(rb_relu_1_1, kernel_size=3, filters=32, strides=2, padding="same",
                                           kernel_initializer=tf.keras.initializers.he_normal(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                           name="rb_conv_1_1")

            if training:
                rb_bn_1_2 = tf.layers.batch_normalization(rb_conv_1_1, training=training)
            else:
                rb_bn_1_2 = rb_conv_1_1
            rb_relu_1_2 = tf.nn.relu(rb_bn_1_2, name="rb_relu_1_2")
            rb_conv_1_2 = tf.layers.conv2d(rb_relu_1_2, kernel_size=3, filters=32, padding="same",
                                           kernel_initializer=tf.keras.initializers.he_normal(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                           name="rb_conv_1_2")

            rb_conv_1_3 = tf.layers.conv2d(max_pool_1, kernel_size=1, filters=32, strides=2, padding="same",
                                           name="rb_conv_1_3")

            rb_out_1 = tf.add(rb_conv_1_2, rb_conv_1_3, name="rb_out_1")

            # flatten
            flat = tf.layers.flatten(rb_out_1)
            flat_relu = tf.nn.relu(flat, name="flat_relu_out")
            flat_dropout = tf.layers.dropout(flat_relu, name="flat_dropout_out")

            # FC_1 - predicting steering angle
            fc_1_steer = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

            # FC_1 - predicting steering angle
            # fc_1_collision = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

            return fc_1_steer


    def model3_h1_d3_n2_padding_same(self, x):
        #CURRENT ARCHITECTURE
        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # f block
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="same",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=8, padding="same",
                                         activation=tf.nn.relu, name="conv_layer_2")

            max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            conv_flat = tf.layers.flatten(max_pool_2)

            # FC_n
            fc_n_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

            # FC_1
            fc_1 = tf.layers.dense(inputs=fc_n_1, units=1, name="fc_layer_out")

            return fc_1

    def model7_h1(self, x):
        '''
        Define model of CNN under the TensorFlow scope "ConvNet".
        The scope is used for better organization and visualization in TensorBoard

        :return: output layer
        '''

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48, 96, 3])

            # define 1st convolutional layer
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # define 2nd convolutional layer
            hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_2")

            #max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            # f 3
            hl_conv_3 = tf.layers.conv2d(hl_conv_2, kernel_size=3, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_3")

            #max_pool_3 = tf.layers.max_pooling2d(hl_conv_3, pool_size=2, strides=2)

            # f 4
            hl_conv_4 = tf.layers.conv2d(hl_conv_3, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_4")

            #max_pool_4 = tf.layers.max_pooling2d(hl_conv_4, pool_size=2, strides=2)

            # f 5
            hl_conv_5 = tf.layers.conv2d(hl_conv_4, kernel_size=5, filters=8, padding="valid",
                                         activation=tf.nn.relu, name="conv_layer_5")

            max_pool_5 = tf.layers.max_pooling2d(hl_conv_5, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            conv_flat = tf.layers.flatten(max_pool_5)

            # FC_n
            # fc_n_1 = tf.layers.dense(inputs=conv_flat, units=512, activation=tf.nn.relu, name="fc_n_layer_1")

            # FC_n
            fc_n_2 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

            # add 2nd fully connected layers to predict the driving commands
            fc_1 = tf.layers.dense(inputs=fc_n_2, units=1, name="fc_layer_out")

            return fc_1