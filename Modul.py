# -*- coding: utf-8 -*-


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import ConvLSTM2D_2
from sklearn.metrics import mean_squared_error

def Modul(x_train, y_train , x_test , y_test , epochs = 10 , Nummber_conv = 2) :
    """epochs : 每一种模型的epochs数量
       Nummber_conv : 每个convlstm中conv的层数
       
       返回最好模型训练集和测试集中的mse值和模型的结构
       """
    sequence_length=15;
    window_size = 15;
    kernel_size_1 = [1,2,3,4,5,6];
    i = 0;
    kernel_size_2_temp = 0;
    for kernel_size_1_temp in kernel_size_1:
        for kernel_size_2_temp in range[1,25]:
            for filter_temp in range[10,15]:
                if Nummber_conv == 1:
                    #build model
                    model =Sequential()
                    model.add(ConvLSTM2D(filters=filter_temp, kernel_size=(kernel_size_1_temp, kernel_size_2_temp),input_shape=(None, window_size, 24, 1), padding='same', return_sequences=True))
                    model.add(keras.layers.BatchNormalization())
                    model.add(ConvLSTM2D(filters=filter_temp, kernel_size=(kernel_size_1_temp, kernel_size_2_temp),padding='same', return_sequences=True))
                    model.add(keras.layers.BatchNormalization())
                    model.add(TimeDistributed(Flatten()))
                    model.add(TimeDistributed(Dense(units = 100, activation = "linear")))
                    model.add(TimeDistributed(Dense(units = 1, activation = "linear")))
                    model.compile(loss='mse', optimizer='Adam')
                    #fit model
                    model.fit(x_train,y_train, batch_size=15, epochs=epochs, validation_split=0.05)
                    #training set
                    y_batch_pred = model.predict(x_train)
                    y_batch_pred = y_batch_pred.reshape(y_batch_pred.shape[0], y_batch_pred.shape[1])
                    y_batch_reshape = y_train.reshape(y_train.shape[0], y_train.shape[1])
                    mse_train = np.sqrt(mean_squared_error(y_batch_pred, y_batch_reshape))
                    #test set
                    y_batch_pred_test = model.predict(x_test)
                    y_batch_pred_test = y_batch_pred_test.reshape(y_batch_pred_test.shape[0],y_batch_pred_test.shape[1])
                    y_batch_pred_last_values = [i[-1] for i in y_batch_pred_test]
                    y_batch_last_values = [i[-1] for i in y_test]
                    mse_test = np.sqrt(mean_squared_error(y_batch_pred_last_values, y_batch_last_values))
                    
                    i = i+1
                    print ('model_', i, ': mse_test = ',mse_test,',','  mse_train = ', mse_train,'------   model_parameter = ', 
                           'kernel_size_1 = ', kernel_size_1_temp, ',','kernel_size_2 = ',kernel_size_2_temp,',','filters = ',filter_temp,',','Nummber_conv = ' ,Nummber_conv)
                    
                    
                    
                elif Nummber_conv ==2:
                    model =Sequential()
                    model.add(ConvLSTM2D_2(filters=filter_temp, kernel_size=(kernel_size_1_temp, kernel_size_2_temp),input_shape=(None, window_size, 24, 1), padding='same', return_sequences=True))
                    model.add(keras.layers.BatchNormalization())
                    model.add(ConvLSTM2D_2(filters=filter_temp, kernel_size=(kernel_size_1_temp, kernel_size_2_temp),padding='same', return_sequences=True))
                    model.add(keras.layers.BatchNormalization())
                    model.add(TimeDistributed(Flatten()))
                    model.add(TimeDistributed(Dense(units = 100, activation = "linear")))
                    model.add(TimeDistributed(Dense(units = 1, activation = "linear")))
                    model.compile(loss='mse', optimizer='Adam')
                    #fit model
                    model.fit(x_train,y_train, batch_size=15, epochs=epochs, validation_split=0.05)
                    #training set
                    y_batch_pred = model.predict(x_train)
                    y_batch_pred = y_batch_pred.reshape(y_batch_pred.shape[0], y_batch_pred.shape[1])
                    y_batch_reshape = y_train.reshape(y_train.shape[0], y_train.shape[1])
                    mse_train = np.sqrt(mean_squared_error(y_batch_pred, y_batch_reshape))
                    #test set
                    y_batch_pred_test = model.predict(x_test)
                    y_batch_pred_test = y_batch_pred_test.reshape(y_batch_pred_test.shape[0],y_batch_pred_test.shape[1])
                    y_batch_pred_last_values = [i[-1] for i in y_batch_pred_test]
                    y_batch_last_values = [i[-1] for i in y_test]
                    mse_test = np.sqrt(mean_squared_error(y_batch_pred_last_values, y_batch_last_values))
                    i = i+1
                    print ('model_', i, ': mse_test = ',mse_test,',','  mse_train = ', mse_train,'------   model_parameter = ', 
                           'kernel_size_1 = ', kernel_size_1_temp, ',','kernel_size_2 = ',kernel_size_2_temp,',','filters = ',filter_temp,',','Nummber_conv = ' ,Nummber_conv)
        
    