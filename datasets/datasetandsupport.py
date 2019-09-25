class DatasetAndSupport(object):

    def quick_load_data(self, patch_shape): raise NotImplementedError

    def load_data(self, patch_shape):raise NotImplementedError

    def evaluate(self, target, pred, cfg):raise NotImplementedError

    def save_results(self, target, pred, cfg): raise NotImplementedError

    def update_checkpoint(self, best_so_far, new_value):raise NotImplementedError