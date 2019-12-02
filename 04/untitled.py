# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:40:55 2019

@author: nandu
"""

# Kamangar, Farhad
# 1000-123-456
# 2019-11_02
# Assignment-04-01


# %tensorflow_version 2.x

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
import os

import matplotlib.pylab as plt

class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network

        """
        self.model=Sequential()
        


    def add_input_layer(self, shape=(2,),name="" ):
        """
         This function adds an input layer to the neural network. If an input layer exist, then this function
         should replcae it with the new input layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """
        
        # self.model.add(Dense(input_shape=shape,name=name))
        self.model.add(InputLayer(input_shape=shape, name=name))

         


    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        
        self.model.add(Dense(units=num_nodes,activation=activation,name=name,trainable=trainable))
        
    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        """
         This function adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
        layer=Conv2D(num_of_filters, kernel_size=kernel_size,padding=padding, strides=strides,
                 activation=activation,
                 name=name,trainable=trainable)
        self.model.add(Conv2D(num_of_filters, kernel_size=kernel_size,padding=padding, strides=strides,
                 activation=activation,
                 name=name,trainable=trainable))
        return layer
        
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This function adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        layer=MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)
        self.model.add(MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name))
        return layer
    
    def append_flatten_layer(self,name=""):
        """
         This function adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        layer=Flatten(name=name)
        self.model.add(Flatten(name=name))
        return layer
        
    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This function sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """
        if(not(layer_number is None)):
         if(layer_number>=0):  
          if(layer_number==0):
             return None
          elif(self.model.get_layer(index=layer_number-1,name=layer_name).count_params()==0):
             return None
          else:
             return self.model.get_layer(index=layer_number-1,name=layer_name).get_weights()[0]
         else:
          if(self.model.get_layer(index=layer_number).count_params()==0):
             return None
          else:
             return self.model.get_layer(index=layer_number).get_weights()[0]
             
        else:
          if(self.model.get_layer(name=layer_name).count_params()==0):
             return None
          else:
             return self.model.get_layer(name=layer_name).get_weights()[0]
            


    def get_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """
        if(not layer_number is None):
            
          if(layer_number==0):
             return None
          elif(self.model.get_layer(index=layer_number-1,name=layer_name).count_params()==0):
             return None
          else:
             return self.model.get_layer(index=layer_number-1,name=layer_name).get_weights()[1]
        else:
          if(self.model.get_layer(name=layer_name).count_params()==0):
             return None
          else:
             return self.model.get_layer(name=layer_name).get_weights()[1]

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
        if(not layer_number is None):
           temp=self.get_biases(layer_number,layer_name)
           W=[weights,temp]
           self.model.get_layer(index=layer_number-1,name=layer_name).set_weights(W)
        else:
           temp=self.get_biases(layer_number,layer_name)
           W=[weights,temp]
           self.model.get_layer(name=layer_name).set_weights(W)
            
            
             
             
       
            
    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        
        if(not layer_number is None):
           temp=self.get_weights_without_biases(layer_number,layer_name)
           W=[temp,biases]
           self.model.get_layer(index=layer_number-1,name=layer_name).set_weights(W)
        else:
           temp=self.get_weights_without_biases(layer_number,layer_name)
           W=[temp,biases]
           self.model.get_layer(name=layer_name).set_weights(W)
             
             
       
    def remove_last_layer(self):
        """
        This function removes a layer from the model.
        :return: removed layer
        """
        layer=self.model.get_layer(index=-1)
        self. model= Sequential(self.model.layers[:-1])
        return layer
        
       # self.model.outputs = [self.model.layers[-1].output]
        

    def load_a_model(self,model_name="",model_file_name=""):
        """
        This function loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
        if(model_name=="VGG16"):
            
              self. model= Sequential(VGG16().layers)

           
        elif(model_name=="VGG19"):
            self. model= Sequential(VGG19().layers)
            
        else:
           self.model= load_model(model_file_name)
        
           
            
        
    def save_model(self,model_file_name=""):
        """
        This function saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """
        self.model.save(model_file_name)
        


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This function sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """

    def set_metric(self,metric):
        """
        This function sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        """
        This function sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        return self.model.predict(X)
        

    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this function returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param X_validation: Array of input validation data
         :param y: Array of desired (target) validation outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
if __name__ == "__main__":

    my_cnn=CNN()
    print(my_cnn)
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    print(my_cnn.model.summary())
    print((my_cnn.model.get_layer(index=-1).count_params()))
    
    
    my_cnn = CNN()
    my_cnn.load_a_model(model_name="VGG19")
    print(my_cnn.model.summary())
    # my_cnn.append_dense_layer(num_nodes=10)
    w=my_cnn.get_weights_without_biases(layer_name="block5_conv4")
    assert w.shape == (3,3,512,512)
    w = my_cnn.get_weights_without_biases(layer_number=-1)
    print(w.shape)
    assert w.shape == (4096,1000)
    my_cnn.append_dense_layer(num_nodes=10)
    path = os.getcwd()
    file_path=os.path.join(path,"my_model.h5")
    my_cnn.save_model(model_file_name=file_path)
    my_cnn.load_a_model(model_name="VGG16")
    w = my_cnn.get_weights_without_biases(layer_name="block4_conv1")
    assert w.shape == (3, 3, 256, 512)
    my_cnn.load_a_model(model_file_name=file_path)
    os.remove(file_path)
    w = my_cnn.get_weights_without_biases(layer_number=-1)
    assert w.shape == (1000,10)
#    m=Sequential()
#    model = VGG19()
#    for layer in model.layers[:-1]:
#              m.add(layer)
#    print(m.summary())
#    my_cnn=CNN()
#    w=m.get_layer(name="block5_conv4").get_weights()[0]
#    print(w.shape)
#    assert w.shape == (3,3,512,512)
    
    
    # my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='linear',name="conv1")
    # print(my_cnn.model.summary())
    weights=my_cnn.get_weights_without_biases(layer_number=0)
    biases=my_cnn.get_biases(layer_number=0)
    print("w0",None if weights is None else weights.shape,type(weights))
    print("b0",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=1)
    biases=my_cnn.get_biases(layer_number=1)
    print("w1",None if weights is None else weights.shape,type(weights))
    print("b1",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=2)
    biases=my_cnn.get_biases(layer_number=2)
    print("w2",None if weights is None else weights.shape,type(weights))
    print("b2",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=3)
    biases=my_cnn.get_biases(layer_number=3)
    print("w3",None if weights is None else weights.shape,type(weights))
    print("b3",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=4)
    biases=my_cnn.get_biases(layer_number=4)
    print("w4",None if weights is None else weights.shape,type(weights))
    print("b4",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_number=5)
    biases = my_cnn.get_biases(layer_number=5)
    print("w5", None if weights is None else weights.shape, type(weights))
    print("b5", None if biases is None else biases.shape, type(biases))

#    weights=my_cnn.get_weights_without_biases(layer_name="input")
#    biases=my_cnn.get_biases(layer_number=0)
#    print("input weights: ",None if weights is None else weights.shape,type(weights))
#    print("input biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv1")
    biases=my_cnn.get_biases(layer_number=1)
    print("conv1 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="pool1")
    biases=my_cnn.get_biases(layer_number=2)
    print("pool1 weights: ",None if weights is None else weights.shape,type(weights))
    print("pool1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv2")
    biases=my_cnn.get_biases(layer_number=3)
    print("conv2 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv2 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="flat1")
    biases=my_cnn.get_biases(layer_number=4)
    print("flat1 weights: ",None if weights is None else weights.shape,type(weights))
    print("flat1 biases: ",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense1")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense1 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense1 biases: ", None if biases is None else biases.shape, type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense2")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense2 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense2 biases: ", None if biases is None else biases.shape, type(biases))
