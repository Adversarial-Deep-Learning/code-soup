import csv
import datetime
import math
import os
import shutil
from collections import Iterable, OrderedDict
from tempfile import NamedTemporaryFile

import torch as th
from tqdm import tqdm


def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs["start_time"] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        logs["final_loss"] = (self.trainer.history.epoch_losses[-1],)
        logs["best_loss"] = (min(self.trainer.history.epoch_losses),)
        logs["stop_time"] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class TQDM(Callback):
    def __init__(self):
        """
        TQDM Progress Bar callback
        """
        self.progbar = None
        super(TQDM, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar is not None:
            self.progbar.close()

    def on_train_begin(self, logs):
        self.train_logs = logs

    def on_epoch_begin(self, epoch, logs=None):
        try:
            self.progbar = tqdm(total=self.train_logs["num_batches"], unit=" batches")
            self.progbar.set_description(
                "Epoch %i/%i" % (epoch + 1, self.train_logs["num_epoch"])
            )
        except:
            pass

    def on_epoch_end(self, epoch, logs=None):
        log_data = {
            key: "%.04f" % value
            for key, value in self.trainer.history.batch_metrics.items()
        }
        for k, v in logs.items():
            if k.endswith("metric"):
                log_data[k.split("_metric")[0]] = "%.02f" % v
        self.progbar.set_postfix(log_data)
        self.progbar.update()
        self.progbar.close()

    def on_batch_begin(self, batch, logs=None):
        self.progbar.update(1)

    def on_batch_end(self, batch, logs=None):
        log_data = {
            key: "%.04f" % value
            for key, value in self.trainer.history.batch_metrics.items()
        }
        for k, v in logs.items():
            if k.endswith("metric"):
                log_data[k.split("_metric")[0]] = "%.02f" % v
        self.progbar.set_postfix(log_data)


class History(Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every SuperModule.
    """

    def __init__(self, model):
        super(History, self).__init__()
        self.samples_seen = 0.0
        self.trainer = model

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {"loss": []}
        self.batch_size = logs["batch_size"]
        self.has_val_data = logs["has_val_data"]
        self.has_regularizers = logs["has_regularizers"]
        if self.has_val_data:
            self.epoch_metrics["val_loss"] = []
        if self.has_regularizers:
            self.epoch_metrics["reg_loss"] = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_metrics = {"loss": 0.0}
        if self.has_regularizers:
            self.batch_metrics["reg_loss"] = 0.0
        self.samples_seen = 0.0

    def on_epoch_end(self, epoch, logs=None):
        for k in self.batch_metrics:
            self.epoch_metrics[k].append(self.batch_metrics[k])

    def on_batch_end(self, batch, logs=None):
        for k in self.batch_metrics:
            self.batch_metrics[k] = (
                self.samples_seen * self.batch_metrics[k] + logs[k] * self.batch_size
            ) / (self.samples_seen + self.batch_size)
        self.samples_seen += self.batch_size

    def __getitem__(self, name):
        return self.epoch_metrics[name]

    def __repr__(self):
        return str(self.epoch_metrics)

    def __str__(self):
        return str(self.epoch_metrics)


class ModelCheckpoint(Callback):
    """
    Model Checkpoint to save model weights during training

    save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        th.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    """

    def __init__(
        self,
        directory,
        filename="ckpt.pth.tar",
        monitor="val_loss",
        save_best_only=False,
        save_weights_only=True,
        max_save=-1,
        verbose=0,
    ):
        """
        Model Checkpoint to save model weights during training

        Arguments
        ---------
        file : string
            file to which model will be saved.
            It can be written 'filename_{epoch}_{loss}' and those
            values will be filled in before saving.
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        save_best_only : boolean
            whether to only save if monitored value has improved
        save_weight_only : boolean
            whether to save entire model or just weights
            NOTE: only `True` is supported at the moment
        max_save : integer > 0 or -1
            the max number of models to save. Older model checkpoints
            will be overwritten if necessary. Set equal to -1 to have
            no limit
        verbose : integer in {0, 1}
            verbosity
        """
        if directory.startswith("~"):
            directory = os.path.expanduser(directory)
        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.max_save = max_save
        self.verbose = verbose

        if self.max_save > 0:
            self.old_files = []

        # mode = 'min' only supported
        self.best_loss = math.inf
        super(ModelCheckpoint, self).__init__()

    def save_checkpoint(self, state, is_best=False):
        th.save(state, self.file)
        if is_best:
            shutil.copyfile(self.file, "model_best.pth.tar")

    def on_epoch_end(self, epoch, logs=None):
        file = self.file.format(
            epoch="%03i" % (epoch + 1), loss="%0.4f" % logs[self.monitor]
        )
        if self.save_best_only:
            current_loss = logs.get(self.monitor)
            if current_loss is None:
                pass
            else:
                if current_loss < self.best_loss:
                    if self.verbose > 0:
                        print(
                            "\nEpoch %i: improved from %0.4f to %0.4f saving model to %s"
                            % (epoch + 1, self.best_loss, current_loss, file)
                        )
                    self.best_loss = current_loss
                    # if self.save_weights_only:
                    self.trainer.save_state_dict(file)
                    # else:
                    #    self.save_checkpoint({
                    #            'epoch': epoch + 1,
                    #            #'arch': args.arch,
                    #            'state_dict': self.trainer.state_dict(),
                    #            #'best_prec1': best_prec1,
                    #            'optimizer' : self.trainer.optimizer.state_dict(),
                    #            #'loss':{},
                    #            #'regularizers':{},
                    #            #'constraints':{},
                    #            #'initializers':{},
                    #            #'metrics':{},
                    #            #'val_loss':{}
                    #        })
                    if self.max_save > 0:
                        if len(self.old_files) == self.max_save:
                            try:
                                os.remove(self.old_files[0])
                            except:
                                pass
                            self.old_files = self.old_files[1:]
                        self.old_files.append(file)
        else:
            if self.verbose > 0:
                print("\nEpoch %i: saving model to %s" % (epoch + 1, file))
            self.trainer.save_state_dict(file)
            if self.max_save > 0:
                if len(self.old_files) == self.max_save:
                    try:
                        os.remove(self.old_files[0])
                    except:
                        pass
                    self.old_files = self.old_files[1:]
                self.old_files.append(file)


class CSVLogger(Callback):
    """
    Logs epoch-level metrics to a CSV file
    """

    def __init__(self, file, separator=",", append=False):
        """
        Logs epoch-level metrics to a CSV file

        Arguments
        ---------
        file : string
            path to csv file
        separator : string
            delimiter for file
        apped : boolean
            whether to append result to existing file or make new file
        """
        self.file = file
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.file):
                with open(self.file) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.file, "a")
        else:
            self.csv_file = open(self.file, "w")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        RK = {"num_batches", "num_epoch"}

        def handle_value(k):
            is_zero_dim_tensor = isinstance(k, th.Tensor) and k.dim() == 0
            if isinstance(k, Iterable) and not is_zero_dim_tensor:
                return '"[%s]"' % (", ".join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=["epoch"] + [k for k in self.keys if k not in RK],
                dialect=CustomDialect,
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({"epoch": epoch})
        row_dict.update(
            (key, handle_value(logs[key])) for key in self.keys if key not in RK
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class ExperimentLogger(Callback):
    def __init__(
        self,
        directory,
        filename="Experiment_Logger.csv",
        save_prefix="Model_",
        separator=",",
        append=True,
    ):

        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename)
        self.save_prefix = save_prefix
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(ExperimentLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            open_type = "a"
        else:
            open_type = "w"

        # if append is True, find whether the file already has header
        num_lines = 0
        if self.append:
            if os.path.exists(self.file):
                with open(self.file) as f:
                    for num_lines, l in enumerate(f):
                        pass
                    # if header exists, DONT append header again
                with open(self.file) as f:
                    self.append_header = not bool(len(f.readline()))

        model_idx = num_lines
        REJECT_KEYS = {"has_validation_data"}
        MODEL_NAME = self.save_prefix + str(
            model_idx
        )  # figure out how to get model name
        self.row_dict = OrderedDict({"model": MODEL_NAME})
        self.keys = sorted(logs.keys())
        for k in self.keys:
            if k not in REJECT_KEYS:
                self.row_dict[k] = logs[k]

        class CustomDialect(csv.excel):
            delimiter = self.sep

        with open(self.file, open_type) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["model"] + [k for k in self.keys if k not in REJECT_KEYS],
                dialect=CustomDialect,
            )
            if self.append_header:
                writer.writeheader()

            writer.writerow(self.row_dict)
            csv_file.flush()

    def on_train_end(self, logs=None):
        REJECT_KEYS = {"has_validation_data"}
        row_dict = self.row_dict

        class CustomDialect(csv.excel):
            delimiter = self.sep

        self.keys = self.keys
        temp_file = NamedTemporaryFile(delete=False, mode="w")
        with open(self.file, "r") as csv_file, temp_file:
            reader = csv.DictReader(
                csv_file,
                fieldnames=["model"] + [k for k in self.keys if k not in REJECT_KEYS],
                dialect=CustomDialect,
            )
            writer = csv.DictWriter(
                temp_file,
                fieldnames=["model"] + [k for k in self.keys if k not in REJECT_KEYS],
                dialect=CustomDialect,
            )
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    # re-write header with on_train_end's metrics
                    pass
                if row["model"] == self.row_dict["model"]:
                    writer.writerow(row_dict)
                else:
                    writer.writerow(row)
        shutil.move(temp_file.name, self.file)
