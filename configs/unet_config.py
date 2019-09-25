from configs.config import Config

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
    cfg.model_name = 'unet'
    cfg.first_layer_channels = 32
    cfg.num_input_channel = 1
    cfg.step_count = 4

    ''' Training '''
    cfg.numb_of_epochs = 25000
    cfg.eval_every = 1
    cfg.lamda_ce = 1
    cfg.batch_size = 1
    cfg.learning_rate = 1e-4

    ''' Priors '''
    cfg.priors = None
    cfg.augmentation_shift_range = 15

    ''' Save at '''
    cfg.save_path = '/cvlabdata1/cvlab/datasets_udaranga/experiments/miccai2019/'
    cfg.save_dir_prefix = 'Experiment_'

    return cfg
