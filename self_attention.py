# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import tensorflow as tf
from tensorflow.keras import layers



class self_attention(layers.Layer):
    """
    input_shape should be a 5 dimensional tensor
    input_shape = batch_size * timestep * row * feature * channels
    feature should be 1 after the Conv_LSTM
    the input will have to rehape first 
    shape = batch_size * [timestep*row*feature] * channels
    [timestep*row*feature] ist the time feature
    output_shape is 3D tensor
    this layer apply the attention mechanismus on channels with time feature
    
    if key_value = True
        the key and value are the same
        key = value = input
    if key_value = False
        key = input x key_weight
        value = input x value_weight
    query = input x query_weights
    score = dot_product(query, key)
    Calculate softmax(score)
    alignment_vectors = value * score
    output = sum(alignment_vectors of all input)
    output is 3D
    
    """

    def __init__(self, 
                 key_value = False,):
        super(self_attention, self).__init__()
        self.key_value = key_value

        
    def build(self,input_shape):
        self.key_weight = self.add_weight(shape = (input_shape[1], input_shape[0]),
                                          name = 'key_weight',trainable=True)
        
        self.value_weight = self.add_weight(shape = (input_shape[1], input_shape[0]),
                                          name = 'value_weight',trainable=True)
        
        self.query_weight = self.add_weight(shape = (input_shape[1], input_shape[0]),
                                          name = 'query_weight',trainable=True)
        
        
    def call(self, inputs):
        query = tf.matmul(inputs,self.query_weight)
        
        value = tf.matmul(inputs,self.value_weight)
        
        key = tf.matmul(inputs,self.key_weight)
        
        key_r = tf.transpose(key)
        
        score = tf.matmul(query,key_r)
        
        score_soft = tf.nn.softmax(score, axis = 1)
        
        score_r = tf.transpose(score_soft)
        
        outputs = value[:,None]*score_r[:,:,None]
        
        output = tf.keras.backend.sum(outputs,axis=0)
        
        return output

        
        
        

