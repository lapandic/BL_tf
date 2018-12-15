#!/usr/bin/env python

import tensorflow as tf

def get_model(history,steps_ahead,arch_num):

    models_bs_1_fs_1 = [model1_h1_d1_n1, model2_h1_d2_n1, model3_h1_d3_n1, model_dronet_1rb, model3_h1_d3_n2, model3_h1_d3_n3]

    models_bs_2_fs_1 = [model_bs_2_fs_1_d3, model_bs_2_fs_1_rbs]

    models_bs_3_fs_1 = [model_bs_3_fs_1_d3, model_bs_3_fs_1_rbs]

    models_bs_1_fs_5 = [model_bs_1_fs_5_d3, model_bs_1_fs_5_d4]

    models_bs_2_fs_5 = [model_bs_2_fs_5_d3, model_bs_2_fs_5_d4]

    models_bs_3_fs_5 = [model_bs_3_fs_5_d3, model_bs_3_fs_5_d4]

    models_bs_4_fs_5 = [model_bs_4_fs_5_d3, model_bs_4_fs_5_d4]

    fs_1 = [models_bs_1_fs_1, models_bs_2_fs_1, models_bs_3_fs_1]
    fs_5 = [models_bs_1_fs_5, models_bs_2_fs_5, models_bs_3_fs_5, models_bs_4_fs_5]

    if steps_ahead == 1 and history <= 3:
        if (history > 1 and arch_num < 2) or (history == 1 and arch_num < 6):
            model = fs_1[history][arch_num]
    elif steps_ahead == 5:
        if history < 4 and arch_num < 2:
            model = fs_5[history][arch_num]
    else:
        print("Requested model not implemented!")


    return model


def get_old_model(history,steps_ahead,arch_num):
    models = [model1_h1_d1_n1, model2_h1_d2_n1, model2_h1_d2_n2, model2_h2_d1_n3,
                   model3_h1_d3_n1, model3_h1_d3_n2,
                   model3_h1_d3_n3, model3_h3_d1_n4, model3_h2_d2_n5, model3_h2_d2_n6,
                   model3_h2_d2_n7, model3_h2_d2_n8, model3_h2_d2_n9]

    models_h1 = [model1_h1_d1_n1, model2_h1_d2_n1, model3_h1_d3_n1, model3_h1_d3_n2,
                      model3_h1_d3_n3, model7_h1, model3_h1_d3_3f_2FC1]

    models_h2 = [model2_h2_d1_n3, model3_h2_d2_n5, model3_h2_d2_n6,
                      model3_h2_d2_n7, model3_h2_d2_n8, model3_h2_d2_n9]

    models_h3 = [model3_h3_d1_n4]

    models_dronet = [model_dronet, model_dronet2, model_dronet_1rb]

    if history == 1 and arch_num < 7:
        model = models_h1[arch_num]
    elif history == 2 and arch_num < 6:
        model = models_h2[arch_num]
    elif history == 3:
        model = model3_h3_d1_n4
    elif history == 1 and arch_num > 9:
        model = models_dronet[arch_num - 10]
    else:
        print("Requested model not implemented!")
    return model

def model_bs_1_fs_5_d3(x):
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
        fc_1 = tf.layers.dense(inputs=conv_flat, units=5, name="fc_layer_out")

        return fc_1

def model_bs_1_fs_5_d4(x):
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
        fc_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_layer_1")

        fc_out = tf.layers.dense(inputs=fc_1, units=5, name="fc_layer_out")

        return fc_out

def model_bs_2_fs_5_d3(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

    return fc

def model_bs_2_fs_5_d4(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc_1 = tf.layers.dense(inputs=commands_stack, units=64, activation=tf.nn.relu, name="fc_layer_1")

    fc_out = tf.layers.dense(inputs=fc_1, units=5, name="fc_layer_out")

    return fc_out

def model_bs_3_fs_5_d3(x):
    history = 3
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

    return fc

def model_bs_3_fs_5_d4(x):
    history = 3
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc_1 = tf.layers.dense(inputs=commands_stack, units=64, activation=tf.nn.relu, name="fc_layer_1")

    fc_out = tf.layers.dense(inputs=fc_1, units=5, name="fc_layer_out")

    return fc_out

def model_bs_4_fs_5_d3(x):
    history = 4
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

    return fc

def model_bs_4_fs_5_d4(x):
    history = 4
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc_1 = tf.layers.dense(inputs=commands_stack, units=64, activation=tf.nn.relu, name="fc_layer_1")

    fc_out = tf.layers.dense(inputs=fc_1, units=5, name="fc_layer_out")

    return fc_out

def model_bs_3_fs_1_rbs(x):
    history = 3
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    rb_relu_1_1 = []
    rb_conv_1_1 = []

    rb_relu_1_2 = []
    rb_conv_1_2 = []

    rb_conv_1_3 = []
    rb_out_1 = []
    flat = []
    flat_relu = []
    flat_dropout = []


    for i in range(len(x_array)):
        # first f block NOTE: using linear activation instead of ReLU. NOTE: number of filters!
        hl_conv_1.append(tf.layers.conv2d(x_img, kernel_size=5, filters=32, padding="same",
                                     activation=None, name="conv_layer_1_" + str(i)))

        # NOTE: using pool_size=2 instead of pool_size=3, reason input images in DroNet paper are 200x200x1, we use 48x96x3
        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # Res block 1
        rb_relu_1_1[i] = tf.nn.relu(max_pool_1[i], name="rb_relu_1_1_" + str(i))
        rb_conv_1_1[i] = tf.layers.conv2d(rb_relu_1_1[i], kernel_size=3, filters=32, strides=2, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                       name="rb_conv_1_1_" + str(i))

        rb_relu_1_2[i] = tf.nn.relu(rb_conv_1_1[i], name="rb_relu_1_2_" + str(i))
        rb_conv_1_2[i] = tf.layers.conv2d(rb_relu_1_2[i], kernel_size=3, filters=32, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                       name="rb_conv_1_2_" + str(i))

        rb_conv_1_3[i] = tf.layers.conv2d(rb_conv_1_2[i], kernel_size=1, filters=32, strides=2, padding="same",
                                       name="rb_conv_1_3_" + str(i))

        rb_out_1[i] = tf.add(rb_conv_1_2[i], rb_conv_1_3[i], name="rb_out_1_" + str(i))

        # flatten
        flat[i] = tf.layers.flatten(rb_out_1[i])
        flat_relu[i] = tf.nn.relu(flat[i], name="flat_relu_out_" + str(i))
        flat_dropout[i] = tf.layers.dropout(flat_relu[i], name="flat_dropout_out_" + str(i))

        # FC_1 - predicting steering angle
        # fc_1_steer = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

    commands_stack = tf.concat(flat_dropout, axis=1)

    fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

    return fc


def model_bs_2_fs_1_rbs(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    rb_relu_1_1 = []
    rb_conv_1_1 = []

    rb_relu_1_2 = []
    rb_conv_1_2 = []

    rb_conv_1_3 = []
    rb_out_1 = []
    flat = []
    flat_relu = []
    flat_dropout = []


    for i in range(len(x_array)):
        # first f block NOTE: using linear activation instead of ReLU. NOTE: number of filters!
        hl_conv_1.append(tf.layers.conv2d(x_img, kernel_size=5, filters=32, padding="same",
                                     activation=None, name="conv_layer_1_" + str(i)))

        # NOTE: using pool_size=2 instead of pool_size=3, reason input images in DroNet paper are 200x200x1, we use 48x96x3
        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # Res block 1
        rb_relu_1_1[i] = tf.nn.relu(max_pool_1[i], name="rb_relu_1_1_" + str(i))
        rb_conv_1_1[i] = tf.layers.conv2d(rb_relu_1_1[i], kernel_size=3, filters=32, strides=2, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                       name="rb_conv_1_1_" + str(i))

        rb_relu_1_2[i] = tf.nn.relu(rb_conv_1_1[i], name="rb_relu_1_2_" + str(i))
        rb_conv_1_2[i] = tf.layers.conv2d(rb_relu_1_2[i], kernel_size=3, filters=32, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                       name="rb_conv_1_2_" + str(i))

        rb_conv_1_3[i] = tf.layers.conv2d(rb_conv_1_2[i], kernel_size=1, filters=32, strides=2, padding="same",
                                       name="rb_conv_1_3_" + str(i))

        rb_out_1[i] = tf.add(rb_conv_1_2[i], rb_conv_1_3[i], name="rb_out_1_" + str(i))

        # flatten
        flat[i] = tf.layers.flatten(rb_out_1[i])
        flat_relu[i] = tf.nn.relu(flat[i], name="flat_relu_out_" + str(i))
        flat_dropout[i] = tf.layers.dropout(flat_relu[i], name="flat_dropout_out_" + str(i))

        # FC_1 - predicting steering angle
        # fc_1_steer = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

    commands_stack = tf.concat(flat_dropout, axis=1)

    fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

    return fc

def model_bs_3_fs_1_d3(x):
    history = 3
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

    return fc

def model_bs_2_fs_1_d3(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

    x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

    hl_conv_1 = []
    max_pool_1 = []
    hl_conv_2 = []
    max_pool_2 = []
    hl_conv_3 = []
    max_pool_3 = []
    conv_flat = []

    for i in range(len(x_array)):
        # define 1st convolutional layer
        hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                          activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

        max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

        # f2
        hl_conv_2.append(tf.layers.conv2d(max_pool_1[i], kernel_size=5, filters=8, padding="valid",
                                     activation=tf.nn.relu, name="conv_layer_2_" + str(i)))

        max_pool_2.append(tf.layers.max_pooling2d(hl_conv_2[i], pool_size=2, strides=2))

        # f3
        hl_conv_3.append(tf.layers.conv2d(max_pool_2[i], kernel_size=5, filters=8, padding="valid",
                                     activation=tf.nn.relu, name="conv_layer_3_" + str(i)))

        max_pool_3.append(tf.layers.max_pooling2d(hl_conv_3[i], pool_size=2, strides=2))

        # flatten tensor to connect it with the fully connected layers
        conv_flat.append(tf.layers.flatten(max_pool_3[i]))

    commands_stack = tf.concat(conv_flat, axis=1)

    fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

    return fc


def model1_h1_d1_n1(x):
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


def model2_h1_d2_n1(x):
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


def model2_h1_d2_n2(x):
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


def model2_h2_d1_n3(x):
    '''
    Define model of CNN under the TensorFlow scope "ConvNet".
    The scope is used for better organization and visualization in TensorBoard

    :return: output layer
    '''
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # define the 4-d tensor expected by TensorFlow
        # [-1: arbitrary num of images, img_height, img_width, num_channels]
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

       x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

        hl_conv_1 = []
        max_pool_1 = []
        conv_flat = []

        for i in range(len(x_array)):
            # define 1st convolutional layer
            hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                              activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

            max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

            # flatten tensor to connect it with the fully connected layers
            conv_flat.append(tf.layers.flatten(max_pool_1[i]))

        commands_stack = tf.concat(conv_flat, axis=1)

        fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

        return fc


def model3_h1_d3_n1(x):
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


def model3_h1_d3_n2(x):
    # CURRENT ARCHITECTURE
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


def model3_h1_d3_n3(x):
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


def model3_h3_d1_n4(x):
    history = 3
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

       x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

        hl_conv_1 = []
        max_pool_1 = []
        conv_flat = []

        for i in range(len(x_array)):
            # f block
            hl_conv_1.append(tf.layers.conv2d(x_array[i], kernel_size=5, filters=2, padding="valid",
                                              activation=tf.nn.relu, name="conv_layer_1_" + str(i)))

            max_pool_1.append(tf.layers.max_pooling2d(hl_conv_1[i], pool_size=2, strides=2))

            conv_flat.append(tf.layers.flatten(max_pool_1[i]))

        commands_stack = tf.concat(conv_flat, axis=1)

        # FC_1
        fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

        return fc


def model3_h2_d2_n5(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

       x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

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

        commands_stack = tf.concat([fc_n_1, conv_flat_2], axis=1)

        # FC_1
        fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

        return fc


def model3_h2_d2_n6(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

       x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

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

        commands_stack = tf.concat([conv_flat_1, conv_flat_2], axis=1)

        # FC_1
        fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

        return fc


def model3_h2_d2_n7(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

       x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

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

        commands_stack = tf.concat([conv_flat_1, conv_flat_2], axis=1)

        # FC_1
        fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

        return fc


def model3_h2_d2_n8(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

       x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

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

        commands_stack = tf.concat([conv_flat_1, fc_n_1], axis=1)

        # FC_1
        fc = tf.layers.dense(inputs=commands_stack, units=1, name="fc_layer_out")

        return fc


def model3_h2_d2_n9(x):
    history = 2
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x_img = tf.reshape(x, [-1, 48 * history, 96, 3])

       x_array = tf.split(x_img, num_or_size_splits=history, axis=1)

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

        commands_stack = tf.concat([conv_flat_1, conv_flat_2], axis=1)

        # FC_n
        fc_n_1 = tf.layers.dense(inputs=commands_stack, units=64, activation=tf.nn.relu, name="fc_n_layer_1")

        # FC_1
        fc = tf.layers.dense(inputs=fc_n_1, units=1, name="fc_layer_out")

        return fc


def model_dronet(x, training):
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
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_1_1")

        if training:
            rb_bn_1_2 = tf.layers.batch_normalization(rb_conv_1_1, training=training)
        else:
            rb_bn_1_2 = rb_conv_1_1
        rb_relu_1_2 = tf.nn.relu(rb_bn_1_2, name="rb_relu_1_2")
        rb_conv_1_2 = tf.layers.conv2d(rb_relu_1_2, kernel_size=3, filters=32, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_1_2")

        rb_conv_1_3 = tf.layers.conv2d(max_pool_1, kernel_size=1, filters=32, strides=2, padding="same",
                                       name="rb_conv_1_3")

        rb_out_1 = tf.add(rb_conv_1_2, rb_conv_1_3, name="rb_out_1")

        # Res block 2
        if training:
            rb_bn_2_1 = tf.layers.batch_normalization(rb_out_1, training=training)
        else:
            rb_bn_2_1 = rb_out_1
        rb_relu_2_1 = tf.nn.relu(rb_bn_2_1, name="rb_relu_2_1")
        rb_conv_2_1 = tf.layers.conv2d(rb_relu_2_1, kernel_size=3, filters=64, strides=2, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_2_1")

        if training:
            rb_bn_2_2 = tf.layers.batch_normalization(rb_conv_2_1, training=training)
        else:
            rb_bn_2_2 = rb_conv_2_1
        rb_relu_2_2 = tf.nn.relu(rb_bn_2_2, name="rb_relu_2_2")
        rb_conv_2_2 = tf.layers.conv2d(rb_relu_2_2, kernel_size=3, filters=64, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_2_2")

        rb_conv_2_3 = tf.layers.conv2d(rb_out_1, kernel_size=1, filters=64, strides=2, padding="same",
                                       name="rb_conv_2_3")

        rb_out_2 = tf.add(rb_conv_2_2, rb_conv_2_3, name="rb_out_2")

        # Res block 3
        if training:
            rb_bn_3_1 = tf.layers.batch_normalization(rb_out_2, training=training)
        else:
            rb_bn_3_1 = rb_out_2
        rb_relu_3_1 = tf.nn.relu(rb_bn_3_1, name="rb_relu_2_1")
        rb_conv_3_1 = tf.layers.conv2d(rb_relu_3_1, kernel_size=3, filters=128, strides=2, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_3_1")

        if training:
            rb_bn_3_2 = tf.layers.batch_normalization(rb_conv_3_1, training=training)
        else:
            rb_bn_3_2 = rb_conv_3_1
        rb_relu_3_2 = tf.nn.relu(rb_bn_3_2, name="rb_relu_2_2")
        rb_conv_3_2 = tf.layers.conv2d(rb_relu_3_2, kernel_size=3, filters=128, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_3_2")

        rb_conv_3_3 = tf.layers.conv2d(rb_out_2, kernel_size=1, filters=128, strides=2, padding="same",
                                       name="rb_conv_3_3")

        rb_out_3 = tf.add(rb_conv_3_2, rb_conv_3_3, name="rb_out_3")

        # flatten
        flat = tf.layers.flatten(rb_out_3)
        flat_relu = tf.nn.relu(flat, name="flat_relu_out")
        flat_dropout = tf.layers.dropout(flat_relu, name="flat_dropout_out")

        # FC_1 - predicting steering angle
        fc_1_steer = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

        # FC_1 - predicting steering angle
        # fc_1_collision = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

        return fc_1_steer


def model_dronet2(x, training):
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x_img = tf.reshape(x, [-1, 48, 96, 3])

        # first f block NOTE: using linear activation instead of ReLU. NOTE: number of filters!
        hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=32, padding="same",
                                     activation=None, name="conv_layer_1")

        # NOTE: using pool_size=2 instead of pool_size=3, reason input images in DroNet paper are 200x200x1, we use 48x96x3
        max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

        # Res block 1
        rb_bn_1_1 = tf.layers.batch_normalization(max_pool_1, training=training)
        rb_relu_1_1 = tf.nn.relu(rb_bn_1_1, name="rb_relu_1_1")
        rb_conv_1_1 = tf.layers.conv2d(rb_relu_1_1, kernel_size=3, filters=32, strides=2, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_1_1")

        rb_bn_1_2 = tf.layers.batch_normalization(rb_conv_1_1, training=training)
        rb_relu_1_2 = tf.nn.relu(rb_bn_1_2, name="rb_relu_1_2")
        rb_conv_1_2 = tf.layers.conv2d(rb_relu_1_2, kernel_size=3, filters=32, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_1_2")

        rb_conv_1_3 = tf.layers.conv2d(max_pool_1, kernel_size=1, filters=32, strides=2, padding="same",
                                       name="rb_conv_1_3")

        rb_out_1 = tf.add(rb_conv_1_2, rb_conv_1_3, name="rb_out_1")

        # Res block 2
        rb_bn_2_1 = tf.layers.batch_normalization(rb_out_1, training=training)
        rb_relu_2_1 = tf.nn.relu(rb_bn_2_1, name="rb_relu_2_1")
        rb_conv_2_1 = tf.layers.conv2d(rb_relu_2_1, kernel_size=3, filters=64, strides=2, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_2_1")

        rb_bn_2_2 = tf.layers.batch_normalization(rb_conv_2_1, training=training)
        rb_relu_2_2 = tf.nn.relu(rb_bn_2_2, name="rb_relu_2_2")
        rb_conv_2_2 = tf.layers.conv2d(rb_relu_2_2, kernel_size=3, filters=64, padding="same",
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name="rb_conv_2_2")

        rb_conv_2_3 = tf.layers.conv2d(rb_out_1, kernel_size=1, filters=64, strides=2, padding="same",
                                       name="rb_conv_2_3")

        rb_out_2 = tf.add(rb_conv_2_2, rb_conv_2_3, name="rb_out_2")

        # flatten
        flat = tf.layers.flatten(rb_out_2)
        flat_relu = tf.nn.relu(flat, name="flat_relu_out")
        flat_dropout = tf.layers.dropout(flat_relu, name="flat_dropout_out")

        # FC_1 - predicting steering angle
        fc_1_steer = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

        # FC_1 - predicting steering angle
        # fc_1_collision = tf.layers.dense(inputs=flat_dropout, units=1, name="fc_layer_out")

        return fc_1_steer


def model_dronet_1rb(x, training):
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


def model3_h1_d3_n2_padding_same(x):
    # CURRENT ARCHITECTURE
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


def model7_h1(x):
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

        # max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

        # f 3
        hl_conv_3 = tf.layers.conv2d(hl_conv_2, kernel_size=3, filters=8, padding="valid",
                                     activation=tf.nn.relu, name="conv_layer_3")

        # max_pool_3 = tf.layers.max_pooling2d(hl_conv_3, pool_size=2, strides=2)

        # f 4
        hl_conv_4 = tf.layers.conv2d(hl_conv_3, kernel_size=5, filters=8, padding="valid",
                                     activation=tf.nn.relu, name="conv_layer_4")

        # max_pool_4 = tf.layers.max_pooling2d(hl_conv_4, pool_size=2, strides=2)

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


def model3_h1_d3_3f_2FC1(x):
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
        fc_1 = tf.layers.dense(inputs=conv_flat, units=1, name="fc_layer_1")

        fc_2 = tf.layers.dense(inputs=conv_flat, units=1, name="fc_layer_2")

        fc_out = tf.concat([fc_1, fc_2], axis=1, name="fc_layer_out")

        return fc_out