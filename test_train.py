from utils import parse_args
from utils import dir_utils
import gin
from subprocess import call
import pickle
import random
#import tensorflow as tf
import keras
import numpy as np
import load_data

def getIndex(data, agent):
    split_index = int(0.9*len(data[agent]))
    return split_index

    
def makeModel(input_size, output_size):  
    '''model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape = (input_size)),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(output_size, activation = 'softmax')
    ])'''
    inputs = keras.layers.Input(shape=(input_size,))
    '''x = keras.layers.Dense(1024, activation = 'relu')(inputs)
    x =  keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation = 'relu')(x)
    x = keras.layers.Dropout(0.1)(x)'''
    #x = keras.layers.Dense(256, activation = 'relu')(x)
    x = keras.layers.Dense(256, activation = 'relu')(inputs)
    outputs = keras.layers.Dense(output_size, activation = 'sigmoid')(x)
    model = keras.models.Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics= ['accuracy'])
    return model

def main(args):
    data = load_data.main(args) 
    input_size = 0
    output_size = 0
    for agent in data.test_data:
        for game in data.test_data[agent]:
            input_size = len(game[0][1])
            output_size = len(game[1][1])
            break 
    #print(input_size)
    #print(output_size)
    model = makeModel(input_size, output_size)
    observations = []
    actions = []
    for agent in data.train_data:
        for game in data.train_data[agent]:
            for step in game[0]:
                observations.append(step)
            for step in game[1]:
                actions.append(step)
    model.fit(np.array(observations), np.array(actions), validation_split = 0.1, epochs = 10)
    for agent in data.test_data:
        observations_test = []
        actions_test = []
        for game in data.test_data[agent]:
            for step in game[0]:
                observations_test.append(step)
            for step in game[1]:
                actions_test.append(step)
        predicted_output = model.evaluate(np.array(observations_test), np.array(actions_test))
        print(predicted_output)
        #model.evaluate()


if __name__ == "__main__":
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)
    main(args)
    
    
