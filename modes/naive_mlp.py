# This mode is for the full model with all the bells and whistles.

import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.activations import softmax, relu
from keras.models import Model
from keras.initializers import glorot_normal, Constant
from keras import regularizers
import gin


@gin.configurable
def build_model(args, cfg={}):
    input_obs = Input(shape=(cfg['input_obs_dim'],))

    h1 = my_dense(input_obs, cfg['h1_dim'], name='h1')
    h2 = my_dense(h1, cfg['h2_dim'], name='h2')    
    h3 = my_dense(h2, cfg['h3_dim'], name='h3')
    
    output = my_dense(h3, cfg['output_dim'], name='output', activation='softmax')

    return Model(inputs=[input_obs], outputs=output)

def my_dense(input, output_dim, name, activation='relu'):
    return Dense(units=output_dim, 
                 activation=activation, 
                 kernel_initializer=glorot_normal(seed=0),
                 bias_initializer=Constant(value=0),
                 kernel_regularizer=regularizers.l2(0.0001),
                 name=name)(input)