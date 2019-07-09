# This script is meant as an end to end, data creator, trainer, and evaluater.
# It is set up so that the tasks within can easily be done manually as well,
# by splitting up the tasts into separate scripts/modules.

#import statements
from utils import parse_args, dir_utils, npy_utils
import load_data
import train
#import evaluate
import tensorflow as tf
import keras
import gin
from DataGenerator import DataGenerator
import os
import numpy as np

# Move to utils later on
def bind_gin_external_functions():
    gin.external_configurable(keras.optimizers.Adam, module='keras.optimizers')
    gin.external_configurable(keras.losses.categorical_crossentropy, module='keras.losses')

def main():
    # parse arguments
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)
    args = parse_args.resolve_agentname(args)
    args = dir_utils.resolve_run_directory(args)
    args = dir_utils.resolve_npy_directory(args)

    # bind external functions used by gin
    bind_gin_external_functions()

    # create/load data
    gin.parse_config_file(args.configpath)

    # Assume u have already run the script create_npy_data.py 
    print("----------------- Create Generators -----------------")
    train_generator = DataGenerator(os.path.join(args.datadir, 'train'), args.agentname)
    validation_generator = DataGenerator(os.path.join(args.datadir, 'validation'), args.agentname)
    test_generator = DataGenerator(os.path.join(args.datadir, 'test'), args.agentname)

    # train model/load model
    print("----------------- Start Training -----------------")
    model = train.main(train_generator, validation_generator, args)

    # # evaluate model
    # evaluate.main(data, model, args)


if __name__ == "__main__":
    main()
