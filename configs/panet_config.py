from configs.config import Config
import torch
import numpy as np
from scipy import special

def load_config(dataset_name):

    cfg = Config()

    ''' Experiment '''
    cfg.experiment_idx = 1
    cfg.trial_id = None
    cfg.train_mode = 'train'

    ''' Dataset '''
    cfg.dataset_name = dataset_name
    cfg.set_hint_patch_shape((96, 96, 96))
    cfg.num_classes = 4

    ''' Model '''
    cfg.model_name = 'panet'
    cfg.first_layer_channels = 32
    cfg.num_input_channel = 1
    cfg.step_count = 4
    cfg.theta_param_count = 6
    cfg.pose_latent_feature_dim = 6 * 6 * 6 # for retinal fundus dataset (2D): 18 * 14 , for EM (3D): 6 * 6 * 6
    cfg.pose_latent_feature_channel_count = 8

    ''' Training '''
    cfg.numb_of_epochs = 50000
    cfg.eval_every = 100
    cfg.lamda_ce = 1
    cfg.lamda_angle = 1
    cfg.learning_rate = 1e-4
    cfg.batch_size = 1
    cfg.augmentation_shift_range = 15

    ''' Priors '''


    priors = []
    crop_shape = tuple([int(np.sqrt(2) * i) + cfg.augmentation_shift_range * 2 for i in cfg.hint_patch_shape])

    r = np.linspace(0, crop_shape[1], crop_shape[1], endpoint=False)


    prior = np.zeros(crop_shape, dtype=np.float32)
    prior[:, :, :] = 1 - special.expit(0.5 * (r - crop_shape[1] // 2))[:, None]
    prior = torch.from_numpy(prior).double().cuda()[None]
    priors.append(prior)

    BW = 30;
    band = np.hstack((np.zeros(crop_shape[1] // 2 - BW), np.ones(2 * BW + 1), np.zeros(crop_shape[1] // 2 - BW)))
    prior = np.zeros(crop_shape, dtype=np.float32)
    prior[:, :, :] = np.exp(-2 * ((r - crop_shape[1] // 2) ** 2) / (2 * crop_shape[1]))[:, None]
    prior = torch.from_numpy(prior).double().cuda()[None]
    priors.append(prior)

    prior = np.zeros(crop_shape, dtype=np.float32)
    prior[:, :, :] = special.expit(0.5 * (r - crop_shape[1] // 2))[:, None]
    prior = torch.from_numpy(prior).double().cuda()[None]
    priors.append(prior)

    # prior 4
    D, H, W = crop_shape
    base_grid = torch.zeros((D, H, W, 3))

    w_points = (torch.linspace(0, W - 1, W) if W > 1 else torch.Tensor([-1]))
    h_points = (torch.linspace(0, H - 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
    d_points = (torch.linspace(0, D - 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

    base_grid[:, :, :, 2] = w_points
    base_grid[:, :, :, 1] = h_points
    base_grid[:, :, :, 0] = d_points
    base_grid = base_grid.cuda().float()
    d = base_grid - torch.from_numpy(np.array([D // 2, H // 2, W // 2])).float().cuda()
    prior = 1 - torch.sqrt(torch.sum(d ** 2, 3)) / torch.tensor(np.linalg.norm(crop_shape) // 2).cuda()
    prior = prior[None].double()
    priors.append(prior)

    cfg.priors = (torch.cat(priors, dim=0)[None])
    cfg.prior_channel_count = cfg.priors.shape[1]

    ''' Save at '''
    cfg.save_path = '/cvlabdata1/cvlab/datasets_udaranga/experiments/miccai2019/'
    cfg.save_dir_prefix = 'Experiment_'

    return cfg
