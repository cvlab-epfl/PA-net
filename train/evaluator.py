from utils.utils_common import DataModes, makedirs, append_line, write_lines, blend, crop_indices, blend_cpu
from torch.utils.data import DataLoader
import numpy as np
import torch
from skimage import io

class Evaluator(object):
    def __init__(self, net, optimizer, data, save_path, config, support, train_mode):
        self.data = data
        self.net = net
        self.current_best = None
        self.save_path = save_path + '/best_performance'
        self.optimizer =   optimizer
        self.config = config
        self.support = support
        self.train_mode = train_mode

    def save_model(self, epoch):

        makedirs(self.save_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.save_path + '/model.pth')


    def evaluate(self, epoch, writer=None):
        self.net.eval()
        performences = {}
        for split in [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]:
            dataloader = DataLoader(self.data[split], batch_size=1, shuffle=False)
            performences[split] = self.evaluate_set(dataloader)

            if writer is not None:
                writer.add_scalar(split + '/mean', np.mean(performences[split]), epoch)
                for i in range(1, self.config.num_classes):
                    writer.add_scalar(split + '/class_' + str(i), np.mean(performences[split][:, i-1]), epoch)

        if self.support.update_checkpoint(best_so_far=self.current_best, new_value=performences):
            if self.train_mode == 'train':
                self.save_model(epoch)
            self.save_results(DataLoader(self.data[DataModes.TRAINING], batch_size=1, shuffle=False), epoch, performences[DataModes.TRAINING], self.save_path + '/training_')
            self.save_results(DataLoader(self.data[DataModes.TESTING], batch_size=1, shuffle=False), epoch, performences[DataModes.TESTING], self.save_path + '/testing_')

            self.current_best = performences

        self.net.train()

    def predict(self, data, name):

        if name == 'unet' or name =='vnet':
            x, y = data
            y_hat = self.net(x)
            y_hat = torch.argmax(y_hat, dim=1).cpu()
        elif name == 'panet':
            x, y = data
            y_hat = self.net(x)
            y_hat = torch.argmax(y_hat[0], dim=1).cpu()
            y = y[0]

        return y, y_hat

    def evaluate_set(self, dataloader):
        results = []
        for data in dataloader:
            y, y_hat = self.predict(data, self.config.model_name)
            results.append(self.support.evaluate(y, y_hat, self.config))

        results = np.array(results)
        return results

    def save_results(self, dataloader, epoch, performence, save_path):

        xs = []
        ys = []
        y_hats = []

        for i, data in enumerate(dataloader):
            x, _ = data
            y, y_hat = self.predict(data, self.config.model_name)

            xs.append(x[0, 0])
            ys.append(y[0])
            y_hats.append(y_hat[0])

        if performence is not None:
            performence_mean = 100*np.mean(performence, axis=0)
            summary = ('{}: ' + ', '.join(['{:.2f}' for _ in range(self.config.num_classes-1)])).format(epoch, *performence_mean)

            append_line(save_path + 'summary.txt', summary)
            all_results = [('{}: ' + ', '.join(['{:.2f}' for _ in range(self.config.num_classes-1)])).format(*((i+1,) + tuple(vals))) for i, vals in enumerate(performence)]
            write_lines(save_path + 'all_results.txt', all_results)

        xs = torch.cat(xs, dim=0).cpu()
        ys = torch.cat(ys, dim=0).cpu()
        y_hats = torch.cat(y_hats, dim=0).cpu()

        overlay_y_hat = blend_cpu(xs, y_hats, self.config.num_classes)
        overlay_y = blend_cpu(xs, ys, self.config.num_classes)
        overlay = np.concatenate([overlay_y, overlay_y_hat], axis=2)
        io.imsave(save_path + 'overlay_y_hat.tif', overlay)

