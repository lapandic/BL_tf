#!/usr/bin/env python

import os
import argparse
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
        return graph

def main(model_name):

    # Define the name of your model
    # model_name = 'batch=100,lr=0.0001,optimizer=GDS,epochs=1000,backsteps=5'

    # define the path to the graph from training
    input_graph = os.path.join(os.getcwd(), 'tensorflow_logs', model_name, 'graph', 'graph.pb')

    # define the path in which to save the frozen graph
    output_graph = os.path.join(os.getcwd(), 'tensorflow_logs', model_name, 'frozen_graph', 'frozen_graph.pb')

    # the frozen_graph directory must exist in order to freeze the model
    directory = os.path.join(os.getcwd(), 'tensorflow_logs', model_name, 'frozen_graph')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # define the checkpoint/weights you want to freeze inside the graph
    input_checkpoint = os.path.join(os.getcwd(), 'tensorflow_logs', model_name, 'train-900')

    # define the name of the prediction output node
    # This name can be easily extracted using Tensorboard. In GRAPHS tab of Tensorboard, check the inputs of Loss scope.
    # In this case they are "vel_true" and "ConvNet/fc_layer_2/BiasAdd".The CNN's predictions are provided from the
    # "ConvNet/fc_layer_2/BiasAdd" element, whereas the true omega velocities from the "vel_true". Here we have to define
    # the element which provides the CNN's predictions and thus we defined as output_node_names the "ConvNet/fc_layer_2/BiasAdd".
    output_node_names = "ConvNet/fc_layer_out/BiasAdd"

    # The following settings should remain the same
    input_saver = ""
    input_binary = True
    restore_op_name = 'save/restore_all'
    filename_tensor_name = 'save/Const:0'
    clear_devices = True
    initializer_nodes = ""
    variable_names_blacklist = ""

    # Freeze the graph
    freeze_graph.freeze_graph(
        input_graph,
        input_saver,
        input_binary,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph,
        clear_devices,
        initializer_nodes,
        variable_names_blacklist
    )

    print("The frozen graph is saved in {}.".format(output_graph))

    #g2 = load_graph(output_graph)
    #with g2.as_default():
    #    flops = tf.profiler.profile(g2, options=tf.profiler.ProfileOptionBuilder.float_operation())
    #    print('FLOP after freezing', flops.total_float_ops)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir_name', help='Folder name')
    args = vars(parser.parse_args())
    main(args['dir_name'])
