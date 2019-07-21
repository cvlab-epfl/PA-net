"""
    DO NOT EDIT
"""

class Config(object):
    def __init__(self):
        """ Stores general information necessary to initialize, train and evaluate experiment   """

        ''' Experiment '''
        self.experiment_idx = None
        self.trial_id = None
        self.load_pre_trained = None
        self.train_mode = None

        ''' Save at '''
        self.save_path = None
        self.save_dir_prefix = None

        ''' Dataset '''
        self.dataset_name = None
        self.hint_patch_shape = None

        ''' Model '''
        self.num_input_channel = None
        self.first_layer_channels = None
        self.step_count = None
        self.ndims = None
        self.num_classes = None
        self.theta_param_count = None

        ''' Optimizer '''
        self.learning_rate = None
        self.batch_size = None

        ''' Training '''
        self.numb_of_epochs = None
        self.eval_every = None

        ''' Evaluate '''
        self.eval_function = None
        self.exceed_current_best = None

        self.patch_shape = None


    def set_trial_id(self, trial_id, train_mode=None):
        if trial_id is not None:
            self.train_mode = train_mode #'eval'  # 'train'
        else:
            self.train_mode = None

    def set_hint_patch_shape(self, value):
        self.hint_patch_shape = value
        self.ndims = len(value)

