# This mode is for the full model with all the bells and whistles.

import tensorflow as tf
from keras.layers import Input, Dense, Activation
from keras.models import Model
import modes.network_elements as ne
import gin


@gin.configurable
def build_model(args, cfg={}):
    input_obs = Input(shape=(cfg['input_obs_dim'],))
    # input_act = Input(shape=(cfg['input_act_dim'],))

    h1 = Dense(cfg['h1_dim'], activation=Activation('relu'))(input_obs)
    h2 = Dense(cfg['h2_dim'], activation=Activation('relu'))(h1)
    h3 = Dense(cfg['h3_dim'], activation=Activation('relu'))(h2)
    output = Dense(cfg['output_dim'], activation=None)(h3)

    return Model(inputs=[input_obs], outputs=output)
