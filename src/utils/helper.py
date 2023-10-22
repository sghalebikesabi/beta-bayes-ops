import logging
import numpy as np
from omegaconf import OmegaConf
import os
import random
import wandb
import sys
import torch

import utils


def make_deterministic(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def init(config):
    print(OmegaConf.to_yaml(config))
    run_name = utils.make_run_name(config)
    wandb.init(
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
        config=OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        ),
        name=run_name,
        **config.wandb_args,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    if not os.path.isdir("results"):
        os.mkdir("results")

    if config.logging.get("dir", True):
        results_path = utils.create_date_model_folder("results", run_name, None)
        wandb.log({"results_path": results_path})
    else:
        results_path = None

    if not os.path.isfile("results/attack.csv"):
        with open("results/attack.csv", "a") as f:
            f.write(
                "prefix,seed,task,method,n_rounds,n_obs,n_feats,n_drop,epsilon,private,craft_name,eps_lower,fnr,fpr,acc\n"
            )

    cache_dir = utils.get_project_root() + "/tmp/" + wandb.run.id
    os.makedirs(cache_dir)

    return results_path, cache_dir


def finish(cache_dir):
    wandb.finish()

    filelist = os.listdir(cache_dir)
    for f in filelist:
        os.remove(cache_dir + "/" + f)


def blockPrint():
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__


class ObjectFromDict:
    def __init__(self, d):
        self.__dict__ = d
