import torch
import rocket

from mde.capsules.aligner import *
from mde.capsules.loggers.image_logger import ImageLogger

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)



def make_train(datasets, model, metrics_all, accelerator, loss, optimizer, scheduler, aligner=None, regime=['train','val'], **training_kwargs):
    capsules = list()
    
    # set trainer
    dataset_train =  datasets['train']

    train_looper = rocket.Looper([
            rocket.Dataset(dataset_train, batch_size=training_kwargs['train_batch'], shuffle=True),
            rocket.Module(model, capsules=[
                            rocket.Loss(objective=loss),
                            rocket.Optimizer(optimizer),
                            rocket.Scheduler(scheduler)
                        ]),
            rocket.Checkpointer(output_dir=f"./chkpt/{training_kwargs['exp_name']}",
                                overwrite=training_kwargs['logger_parematers']['overwrite'],
                                save_every=training_kwargs['logger_parematers']['save_every']),
                                #resume_from="./chkpt/149",
                                #resume_capsules=False)
            rocket.Tracker(),
            ImageLogger(save_every=training_kwargs['logger_parematers']['im_log_every'],)
        ], tag="[TRAIN]")

    capsules.append(train_looper)
    
    
    # create evaluation part of train
    if 'val' in regime:
        dataset_val =  datasets['val'] 
        if aligner is not None:
            metrics = [aligner]
        metrics += metrics_all
        
        
        meter = rocket.Meter(capsules=metrics,
                                keys=["target", "predict", "mask"])

        capsules.append(rocket.Looper(capsules=[
                rocket.Dataset(dataset=dataset_val, batch_size=training_kwargs['val_batch']),
                rocket.Module(module=model),
                ImageLogger(save_every=5, mode='val'),
                meter,
                rocket.Tracker(),
            ], grad_enabled=False, tag="[VAL]"))

    

    launcher = rocket.Launcher(capsules=capsules,
                               statefull=False,
                               num_epochs=training_kwargs['num_epochs'],
                               accelerator=accelerator,

    )
    launcher.set_logger(logger)

    

    return launcher