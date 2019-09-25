import os
GPU_index = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

import logging
import torch
import numpy as np
from train.trainer import Trainer
from train.evaluator import Evaluator
from train.networks.network_configs import NetworkConfig
from utils import utils_common
from shutil import copytree, ignore_patterns
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import DataModes

''' Dataset '''
from datasets.CORTEX_EPFL.cortexepfl_voxel import CortexEpfl

''' Config for the experiment '''
from configs import unet_config as config
# from configs import panet_config as config

''' Network '''
from train.networks.unet import UNet as network
# from train.networks.panet import PANet as network

logger = logging.getLogger(__name__)


def main():


    logger.info("Load  Config")
    data_and_support = CortexEpfl()
    cfg = config.load_config(data_and_support.name)

    logger.info("Initialize Experiment")
    trial_path, trial_id, log_msg = init(cfg)
    logger.info('Experiment ID: {}, Trial ID: {}, GPU: {}'.format(cfg.experiment_idx, trial_id, GPU_index))

    logger.info("Network config")
    model_config = NetworkConfig(cfg.step_count, cfg.first_layer_channels, cfg.num_classes, cfg.num_input_channel, True, cfg.ndims, 'same', trial_id, cfg.batch_size, cfg)

    logger.info("Create network")
    classifier = network(model_config)
    classifier.cuda()

    logger.info("Load data")
    cfg.patch_shape = model_config.in_out_shape(cfg.hint_patch_shape)

    data = data_and_support.load_data(cfg)
    loader = DataLoader(data[DataModes.TRAINING], batch_size=classifier.config.batch_size, shuffle=True)
    logger.info("Trainset length: {}".format(loader.__len__()))

    logger.info("Initialize optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=cfg.learning_rate)

    logger.info("Initialize evaluator")
    evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_and_support, cfg.train_mode)

    logger.info("Initialize trainer")
    trainer = Trainer(classifier, loader, optimizer, cfg.numb_of_epochs, cfg.eval_every, trial_path, evaluator, log_msg)

    trainer.train()


def init(cfg):

    save_path = cfg.save_path + cfg.save_dir_prefix + str(cfg.experiment_idx).zfill(3)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    trial_id = (len([dir for dir in os.listdir(save_path) if 'trial' in dir]) + 1) if cfg.trial_id is None else cfg.trial_id

    trial_save_path = save_path + '/trial_' + str(trial_id)

    if not os.path.isdir(trial_save_path):
        os.mkdir(trial_save_path)
        copytree(os.getcwd(), trial_save_path + '/source_code', ignore=ignore_patterns('*.git','*.txt','*.tif', '*.pkl', '*.off', '*.so'))

    log_msg = cfg.save_dir_prefix + str(cfg.experiment_idx).zfill(3) + ', trial ' + str(trial_id) + ', GPU ' + str(GPU_index)



    # setting seed so that experiments are reproducible
    seed = trial_id

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation
    torch.backends.cudnn.benchmark = False  # if it is true, algorithms get stochastic
    torch.backends.cudnn.deterministic = True  # force cudnn to use diterministic algorithms

    return trial_save_path, trial_id, log_msg

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    utils_common.config_logger("/dev/null")
    main()