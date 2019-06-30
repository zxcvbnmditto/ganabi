from utils import parse_args
import importlib
import load_data
import gin
# import keras


@gin.configurable
class Trainer(object):
    @gin.configurable
    def __init__(self, args,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 batch_size=None,
                 epochs=None):

        self.optimizer = optimizer
        self.optimizer.get_config()
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs

def main(train_data_generator, val_data_generator, args):
    trainer = Trainer(args) # gin configured

    #FIXME: combine into one line once stuff works
    mode_module = importlib.import_module(args.mode)                          
    model = mode_module.build_model(args)

    model.compile(
            optimizer = trainer.optimizer,
            loss = trainer.loss,
            metrics = trainer.metrics)

    tr_history = model.fit_generator(
            generator = train_data_generator,
            verbose = 2, # one line per epoch
            epochs = trainer.epochs,
            validation_data = val_data_generator,
            shuffle = True)
              
    return model

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    data = load_data.main(args)
    main(data, args)
