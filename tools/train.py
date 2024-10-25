import mde
import hydra
from omegaconf import OmegaConf
from accelerate import DistributedDataParallelKwargs
import accelerate



@hydra.main(version_base=None, config_path="../config", config_name="train.yaml")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))

    accelerator = hydra.utils.instantiate(cfg.accelerator)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator.ddp_handler = ddp_kwargs
    

    # init modelsls
    model = hydra.utils.instantiate(cfg.launch.model)
    params = model.get_parameters()
    
    # set optimizer and scheduler
    optimizer = hydra.utils.instantiate(cfg.training_configs.optimizer, params=params)
    scheduler = hydra.utils.instantiate(cfg.training_configs.scheduler, optimizer=optimizer)
    
    # set loss
    loss = hydra.utils.instantiate(cfg.training_configs.loss)

    datasets =  {'train': hydra.utils.instantiate(cfg.launch.dataset_train),
                 'val': hydra.utils.instantiate(cfg.launch.dataset_val)}

    aligner = hydra.utils.instantiate(cfg.launch.aligner)
    
    metrics = []
    for metric in cfg.launch.metrics:
        metric = hydra.utils.instantiate(metric)
        metrics.append(metric)

    
    launcher = mde.launchers.make_train(accelerator=accelerator,
                                        datasets=datasets,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        loss=loss,
                                        aligner=aligner,
                                        metrics_all=metrics,
                                        **cfg.training_configs.training_parameters,
                                        )
    launcher.launch()


if __name__ == '__main__':
    run()
