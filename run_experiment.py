# This script is meant as an end to end, data creator, trainer, and evaluater.
# It is set up so that the tasks within can easily be done manually as well,
# by splitting up the tasts into separate scripts/modules.

#import statements
from utils import parse_args, dir_utils, npy_utils
import load_data
import train
#import evaluate
import keras
import gin
from DataGenerator import DataGeneratorMananger
import sys
sys.path.append('./modes') # Temporary Fix for config.gin

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
    if args.newnpy:
        dir_utils.remove_npy_data(args)

    # bind external functions used by gin
    bind_gin_external_functions()
    gin.parse_config_file(args.configpath)

    print("----------------- Create Generators -----------------")
    loader = DataGeneratorMananger(args.datadir, args.agentname)

    # train model/load model
    print("----------------- Start Training -----------------")
    model = train.main(loader, args)

    # # evaluate model
    # evaluate.main(data, model, args)


if __name__ == "__main__":
    main()
