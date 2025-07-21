import yaml
import time
import logging
import os

import optax
import torchvision
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf
import jax.numpy as jnp

# global logger
logger = logging.getLogger(__name__)

def create_save_dir(cfg):
    if cfg.uid == "MNIST" or cfg.uid == "MNIST test" or cfg.uid == "MNIST S5":
        save_dir = f"{cfg.logging.save_path}/{cfg.uid}/{cfg.model}/{time.strftime("%c", time.gmtime(time.time()))}"
    else: 
        save_dir = f"{cfg.logging.save_path}/{cfg.uid}/{cfg.env_name}/{cfg.model}/{time.strftime("%c", time.gmtime(time.time()))}"

    try: 
        os.makedirs(save_dir)
    except OSError:
        logger.error(f"Directory '{save_dir}' already exists.")
    except PermissionError:
        logger.error(f"Permission denied: Unable to create '{save_dir}'.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    return save_dir

def get_save_dir(cfg):
    return  f"{cfg.logging.save_path}/{cfg.uid}/{cfg.model} {cfg.env_name}"

def log_metrics(metric_dict, save_dir):
    try:
        os.mkdir(f"{save_dir}/metrics")
    except OSError:
        logger.error(f"Directory '{save_dir}/metrics' already exists.")
    except PermissionError:
        logger.error(f"Permission denied: Unable to create '{save_dir}/metrics'.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    for key, value in metric_dict.items():
        jnp.save(f"{save_dir}/metrics/{key}", value)

def load_yaml_config(file) -> dict:
    # Read in yaml file and convert to omegaconf
    with open(file) as f:
        yaml_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(yaml_dict)
    return cfg

def cfg2dict(cfg):
    """
    Recursively convert OmegaConf to vanilla dict
    :param cfg:
    :return:
    """
    return OmegaConf.to_container(cfg, resolve=True)

def initialise_MNIST_dataloaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE):

    normalise_sequentialise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5,), (0.5,)),
            torchvision.transforms.Lambda(lambda x: x.view(-1))
        ]
    )

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=normalise_sequentialise_data
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=normalise_sequentialise_data
    )

    trainloader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True
    )

    return trainloader, testloader