# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:28:11 2020

@author: gaoyu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:11:51 2020

@author: gaoyu
"""
import hpbandster
from collections import Counter
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np

import hpbandster
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers.convolutional import Conv3D
#from tensorflow.keras.layers.convolutional_recurrent import ConvLSTM2D
import tensorflow.keras as keras
from ConvLSTM2D_2 import ConvLSTM2D_2

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker



class KerasWorker(Worker):
    def __init__(self,N_train=19000, N_valid=1000, **kwargs):
            super().__init__(**kwargs)
            self.train = pd.read_csv('./Train_resampeled.csv',sep = ',')
            self.vali = pd.read_csv('./vali.csv',sep = ',')
            #test = pd.read_csv('./test.csv',sep = ',')
            
            sequence_length = 20
            window_size = 109
            #kernel_size = 3
            sliding = 10
            
            # batch_generator
            
            def batch_generator_1(data, sequence_length=15, window_size = 10, sliding = 10):
                #batch_number = data.shape[0]/(sequence_length+window_size-1)
                #feature_number = data.shape[1]-1
                window_number = int((data.shape[0]-window_size)/sliding)
                temp_shape = (window_number, window_size, data.shape[1])
                temp = np.zeros(shape=temp_shape, dtype=np.float32)
                for window in range(window_number):
                    temp[window] = data.iloc[window*sliding:window*sliding+window_size,:]
                    
                return temp
            
            def batch_generator(temp,sequence_length,window_size):
                feature_number = temp.shape[2]-1
                batch_number = int(temp.shape[0]/sequence_length)
                x_shape = (batch_number, sequence_length, window_size, feature_number)
                x_batch = np.zeros(shape=x_shape, dtype=np.float32)
                y_shape = (batch_number, sequence_length, window_size)
                y_batch = np.zeros(shape=y_shape, dtype=np.float32)
                for Batch in range(batch_number):
                    for seq in range(sequence_length):
                        x_batch[Batch,seq] = temp[Batch*sequence_length+seq,:,:-1]
                        #y_batch[Batch] = max(temp[Batch*sequence_length:Batch*sequence_length-1,:,-1],key=temp[Batch*sequence_length:Batch*sequence_length-1,:,-1].tolist().count)
                        y_batch[Batch,seq] = temp[Batch*sequence_length+seq,:,-1]
                return x_batch, y_batch
            
            def y_batch_generator(y_batch,sequence_length=15,num_classes=7):
                batch_number = y_batch.shape[0]
                y_batch_n = y_batch
                y_shape = (batch_number, sequence_length)
                y_new = np.zeros(shape=y_shape, dtype=np.float32)
                y_batch_new_shape = (batch_number,sequence_length,num_classes)
                y_batch_new = np.zeros(shape=y_batch_new_shape, dtype=np.float32)
                y_temp = y_batch_n.tolist()
                for batch in range(batch_number):
                    for seq in range(sequence_length):
                        temp_1 = list(Counter(y_temp[batch][seq]).most_common(1))
                        y_new[batch,seq]=list(temp_1[0])[0]
                    y_batch_new[batch] = tf.keras.utils.to_categorical(y_new[batch],num_classes=num_classes)
                    
                return y_batch_new
                    
                
            self.train = batch_generator_1(self.train,sequence_length=sequence_length, window_size=window_size,sliding = sliding)
            self.x_batch, self.y_batch= batch_generator(self.train,sequence_length=sequence_length, window_size=window_size)
            self.x_batch = np.expand_dims(self.x_batch, axis=4)
            self.y_batch = y_batch_generator(self.y_batch,sequence_length = sequence_length)
            
            ##############################
            self.vali = batch_generator_1(self.vali,sequence_length=sequence_length, window_size=window_size)
            self.x_batch_vali, self.y_batch_vali= batch_generator(self.vali,sequence_length=sequence_length, window_size=window_size)
            self.x_batch_vali = np.expand_dims(self.x_batch_vali, axis=4)
            self.y_batch_vali = y_batch_generator(self.y_batch_vali,sequence_length=sequence_length)
            self.y_batch_vali = np.array(self.y_batch_vali)
            
            self.batch_size = 64
            self.num_classes = 7

            # the data, split between train and test sets
            self.x_train = self.x_batch
            self.y_train = self.y_batch



            self.x_train, self.y_train = self.x_train[:N_train], self.y_train[:N_train]
            self.x_validation, self.y_validation = self.x_batch_vali[-N_valid:], self.y_batch_vali[-N_valid:]
            
            


            self.input_shape = ( self.x_batch.shape[1], self.x_batch.shape[2],self.x_batch.shape[3],self.x_batch.shape[4])
            

    def compute(self, config, budget, working_directory, *args, **kwargs):
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """

            model = Sequential()

            model.add(ConvLSTM2D_2(config['num_filters_1'], kernel_size=(3,self.x_batch.shape[3]),activation='relu',input_shape=self.input_shape,return_sequences = True))
            model.add(keras.layers.BatchNormalization())

            if config['num_conv_layers'] > 1:
                    model.add(ConvLSTM2D_2(config['num_filters_2'], kernel_size=(3, 1),
                                                     activation='relu',
                                                     return_sequences = True))
                    model.add(keras.layers.BatchNormalization())

            if config['num_conv_layers'] > 2:
                    model.add(ConvLSTM2D_2(config['num_filters_3'], kernel_size=(3, 1),
                                             activation='relu',
                                             return_sequences = True))
                    model.add(keras.layers.BatchNormalization())

            model.add(Dropout(config['dropout_rate']))
            model.add(TimeDistributed(Flatten()))
            model.add(TimeDistributed(Dense(config['num_fc_units'], activation='relu')))
            model.add(Dropout(config['dropout_rate']))
            model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))


            if config['optimizer'] == 'Adam':
                    optimizer = keras.optimizers.Adam(lr=config['lr'])
            else:
                    optimizer = keras.optimizers.SGD(lr=config['lr'], momentum=config['sgd_momentum'])

            model.compile(loss=keras.losses.categorical_crossentropy,
                                      optimizer=optimizer,
                                      metrics=['accuracy'])

            model.fit(self.x_train, self.y_train,batch_size=self.batch_size, epochs=int(budget),verbose=0,validation_data=(self.x_validation, self.y_validation))

            train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
            val_score = model.evaluate(self.x_validation, self.y_validation, verbose=0)
            

            #import IPython; IPython.embed()
            return ({
                    'loss': 1-val_score[1], # remember: HpBandSter always minimizes!
                    'info': {  'train accuracy': train_score[1],
                                            'validation accuracy': val_score[1],
                                            'number of parameters': model.count_params(),
                                    }

            })  
    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()

            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

            # For demonstration purposes, we add different optimizers as categorical hyperparameters.
            # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
            # SGD has a different parameter 'momentum'.
            optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

            sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

            cs.add_hyperparameters([lr, optimizer, sgd_momentum])



            num_conv_layers =  CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=3, default_value=2)

            num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, default_value=16, log=True)
            num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, default_value=16, log=True)
            num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, default_value=16, log=True)

            cs.add_hyperparameters([num_conv_layers, num_filters_1, num_filters_2, num_filters_3])


            dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
            num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

            cs.add_hyperparameters([dropout_rate, num_fc_units])


            # The hyperparameter sgd_momentum will be used,if the configuration
            # contains 'SGD' as optimizer.
            cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
            cs.add_condition(cond)

            # You can also use inequality conditions:
            cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
            cs.add_condition(cond)

            cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
            cs.add_condition(cond)

            return cs




if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
