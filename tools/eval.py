import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import mde
import hydra
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../config", config_name="eval.yaml")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = hydra.utils.instantiate(cfg)
    launcher = mde.launchers.make_eval(datasets=list(cfg.datasets.values()),
                                    model=cfg.model,
                                    aligner=cfg.aligner,
                                    metrics=list(cfg.metrics.values()),
                                    batch_size=1,
                                    accelerator=cfg.accelerator)
    launcher.launch()


if __name__ == '__main__':
    run()