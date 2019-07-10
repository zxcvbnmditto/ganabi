from utils import parse_args
import importlib
import load_data
import gin
from keras import callbacks

@gin.configurable
class Trainer(object):
    @gin.configurable
    def __init__(self,
        args,
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
        self.callbacks = [
            callbacks.TerminateOnNaN(),
            callbacks.ModelCheckpoint(
            ckpt_filename, 
                monitor='val_acc',
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode='auto',
                period=10
            ),
            callbacks.TensorBoard(
                log_dir=args.resultdir,                               batch_size=self.batch_size,                           update_freq='epoch'
            )
        ]


def main(loader, args):
    trainer = Trainer(args) # gin configured

    train_generator = loader.get('train')
    val_generator = loader.get('validation')

    #FIXME: combine into one line once stuff works
    mode_module = importlib.import_module(args.mode)
    model = mode_module.build_model(args)

    model.compile(
        optimizer=trainer.optimizer,
        loss=trainer.loss,
        metrics=trainer.metrics
    )

    tr_history = model.fit_generator(
        generator = train_generator,
        verbose = 2, # one line per epoch
        epochs = trainer.epochs,
        validation_data = val_generator,
        shuffle = True,
        callbacks = trainer.callbacks
    )

    return model

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    data = load_data.main(args)
    main(data, args)
