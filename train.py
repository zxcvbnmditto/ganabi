from utils import parse_args
import importlib
import load_data
import gin
from keras import callbacks 

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
        ckpt_filename = args.ckptdir + "/" + args.agentname + "-{epoch:02d}-{val_acc:.2f}.hdf5"
        tensorboard_filename = args.resultdir + '/{epoch:02d}-{val_acc:.2f}'
        self.callbacks = [
            callbacks.TerminateOnNaN(),
            callbacks.ModelCheckpoint(
                ckpt_filename, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10),
            # callbacks.EarlyStopping(
            #     monitor='val_acc', min_delta=0.2, patience=20, verbose=1, mode='auto', baseline=0.5, restore_best_weights=False),
            callbacks.TensorBoard(
                log_dir=args.resultdir, batch_size=self.batch_size, update_freq='epoch')
        ]

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
            shuffle = True,
            callbacks = trainer.callbacks)
              
    return model

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    data = load_data.main(args)
    main(data, args)
