''' Dataset '''

import os
GPU_index = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import logging
import torch
import numpy as np
from train.trainer import Trainer
from train.evaluator import Evaluator
from train.networks.network_configs import NetworkConfig
from utils import utils_common
from datasets.load import load_dataset_and_support
from shutil import copytree, ignore_patterns
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import DataModes

logger = logging.getLogger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    f = open(trial_save_path + '/README.txt', 'a')
    f.write(cfg.README)
    f.close()

    # setting seed so that experiments are reproducible
    seed = trial_id

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation
    torch.backends.cudnn.benchmark = False  # if it is true, algorithms get stochastic
    torch.backends.cudnn.deterministic = True  # force cudnn to use diterministic algorithms

    return trial_save_path, trial_id, log_msg

def main():
    ''' Circles dataset, with 50 training experiment 3'''
    from configs import config_unet as config
    dataset_name = 'EM'

    # from configs import config_conv_vae as config
    # dataset_name = 'EM'

    from configs import config_mesh_vae_obsolete as config
    dataset_name = 'EM'

    from configs import config_pca as config
    dataset_name = 'EM'

    # from configs import config_voxel_mesh_net as config
    # dataset_name = 'EM'

    # from configs import config_2 as config
    # dataset_name = 'circles'

    # from configs import config_panet as config
    # dataset_name = 'HIPPO_EM' #HIPPO_EM

    ''' Network '''
    # from train.networks.BaseNetworks.unet import UNet as network
    # from train.networks.panet import PANet as network
    # from train.networks.vae import ConvVAE as network
    # from train.networks.residual_vae import ConvVAE as network

    # from train.networks.AEs.pca import PCANet as network
    # from train.networks.AEs.ae_mesh import AE as network
    # from train.networks.AEs.vae_mesh import VAE as network

    # from train.networks.ModifiedDecoders.meshnet import MeshNet as network
    from train.networks.ModifiedDecoders.surfacenet import SurfaceNet as network
    # from train.networks.ModifiedDecoders.u_meshnet import UMeshNet as network
    # from train.networks.ModifiedDecoders.voxel_basis_net import VoxelBasisNet as network

    cfg = config.load_config(dataset_name)

    # Initialize
    trial_path, trial_id, log_msg = init(cfg)

    logger.info('Experiment ID: {}, Trial ID: {}, GPU: {}'.format(cfg.experiment_idx, trial_id, GPU_index))
    logger.info('Info: {}'.format(cfg.README))

    model_config = NetworkConfig(
        steps=cfg.step_count,
        first_layer_channels=cfg.first_layer_channels,
        ndims=cfg.ndims,
        num_classes=cfg.num_classes,
        num_input_channels=cfg.num_input_channel,
        two_sublayers=True,
        border_mode='same',
        seed=trial_id,
        batch_size=cfg.batch_size,
        config=cfg)

    logger.info("Create network")
    classifier = network(model_config)
    classifier.cuda()

    logger.info("Load data")
    cfg.patch_shape = model_config.in_out_shape(cfg.hint_patch_shape)
    data_and_support = load_dataset_and_support(cfg.dataset_name)


    if cfg.train_mode == 'train':
        data = data_and_support.quick_load_data(cfg)
        loader = DataLoader(data[DataModes.TRAINING], batch_size=classifier.config.batch_size, shuffle=True)
    else:
        # data = data_and_support.load_data(cfg)
        # loader = DataLoader(data['training'], batch_size=classifier.config.batch_size, shuffle=True)
        data = data_and_support.load_data(cfg)
        loader = DataLoader(data, batch_size=1, shuffle=True)

    logger.info("Trainset length: {}".format(loader.__len__()))

    logger.info("Initialize optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=cfg.learning_rate)

    logger.info("Initialize evaluator")
    evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_and_support, cfg.train_mode)

    logger.info("Initialize trainer")
    trainer = Trainer(classifier, loader, optimizer, cfg.numb_of_epochs, cfg.eval_every, trial_path, evaluator, log_msg)

    if cfg.trial_id is not None:
        save_path = trial_path + '/best_performance/model.pth'
        checkpoint = torch.load(save_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0


    if cfg.train_mode == 'train':
        # trainer.do_pca(epoch)
        trainer.train(start_epoch=epoch)
    elif cfg.train_mode == 'eval':
        # evaluator.vae_visualize(loader, epoch)
        evaluator.evaluate(epoch)
        # evaluator.save_results_complete_set(laoder, epoch, None)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    utils_common.config_logger("/dev/null")
    main()