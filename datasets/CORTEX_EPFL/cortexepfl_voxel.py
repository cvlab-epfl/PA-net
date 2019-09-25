import numpy as np
from skimage import io

from evaluate.standard_metrics import jaccard_index
from utils.utils_common import crop, DataModes
from utils import transforms
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
from datasets.datasetandsupport import DatasetAndSupport
import itertools as itr

class Sample:
    def __init__(self, x, y, orientation, center, scale):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.center = center
        self.scale = scale



class CortexVoxelDataset(Dataset):

    def __init__(self, data, cfg, mode):
        self.data = data
        self.cfg = cfg
        self.mode = mode


    def __len__(self):
        return len(self.data)

    def getitem_center(self, idx):
        item = self.data[idx]
        return item.center

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.from_numpy(item.x).cuda()[None]
        y = torch.from_numpy(item.y).cuda().long()

        orientation = torch.from_numpy(item.orientation).float()


        if self.mode == DataModes.TRAINING: # if training do augmentation
            new_orientation = (torch.rand(3) - 0.5) * 2 * self.cfg.augmentation_shift_range
            new_orientation = F.normalize(new_orientation, dim=0)
            q = orientation + new_orientation
            q = F.normalize(q, dim=0)
            theta_rotate = transforms.rot_matrix_from_quaternion(q[None])

            shift = torch.tensor([d/(D//2) for d, D in zip(2 * (torch.rand(3) - 0.5) * self.cfg.augmentation_shift_range, y.shape)])
            theta_shift = transforms.shift(shift)
            theta = theta_rotate @ theta_shift

            x, y = transforms.transform(theta, x, y)
            orientation = new_orientation

            pose = torch.cat((orientation, shift)).cuda()
        else:
            pose = torch.zeros(6).cuda()


        C, D, H, W = x.shape
        center = (D//2, H//2, W//2)

        x = crop(x, (C,) + self.cfg.patch_shape, (0,) + center)
        y = crop(y, self.cfg.patch_shape, center)

        if self.cfg.model_name == 'panet':
            y = [y, pose]

        return x, y


class CortexEpfl(DatasetAndSupport):

    def __init__(self):
        self.name = 'EM'

    def quick_load_data(self, cfg):
        assert cfg.patch_shape == (96, 96, 96), 'Not supported'

        data_root = '/cvlabdata1/cvlab/datasets_udaranga/datasets/3d/graham/'
        class_id = 14
        data_version = 'labels_v' + str(class_id) + '/'

        with open(data_root + data_version + 'labels/pre_computed.pickle', 'rb') as handle:
            data = pickle.load(handle)
        return data

    def load_data(self, cfg):
        '''
        # Change this to load your training data.

        # pre-synaptic neuron   :   1
        # synapse               :   2
        # post-synaptic neuron  :   3
        # background            :   0
        '''

        data_root = '/cvlabdata1/cvlab/datasets_udaranga/datasets/3d/CortexEPFL/'
        num_classes = 4
        path_images = data_root + 'imagestack.tif'
        path_synapse = data_root + 'labels/labels_synapses.tif'
        path_pre_post = data_root + 'labels/labels_pre_post.tif'

        ''' Label information '''
        path_idx = data_root + 'labels/info.npy'
        idx = np.load(path_idx)

        ''' Load data '''
        x = io.imread(path_images)[:200]
        y_synapses = io.imread(path_synapse)
        y_pre_post = io.imread(path_pre_post)

        x = np.float32(x) / 255
        y_synapses = np.int64(y_synapses)

        # Syn at bottom
        temp = np.int64(y_pre_post)
        y_pre_post = np.copy(y_synapses)
        y_pre_post[temp > 0] = temp[temp > 0]

        # method 1: split neurons
        counts = [[0, 12], [12, 24], [24, 36]]

        data = {}

        patch_shape_extended = tuple([int(np.sqrt(2) * i) + 2 * cfg.augmentation_shift_range for i in cfg.patch_shape]) # to allow augmentation and crop

        for i, data_mode in enumerate([DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]):

            samples = []
            for j in range(counts[i][0], counts[i][1]):

                points = np.where(y_synapses == idx[j][1])
                centre = tuple(np.mean(points, axis=1, dtype=np.int64))

                # extract the object of interesete
                y = np.zeros_like(y_pre_post)
                for k, id in enumerate(idx[j][:3]):
                    y[y_pre_post == id] = k + 1

                patch_y = crop(y, patch_shape_extended, centre)
                patch_x = crop(x, patch_shape_extended, centre)

                # Compute orientation

                # First find the Axis
                syn = patch_y == 2
                coords = np.array(np.where(syn)).transpose()
                syn_center = np.flip(np.mean(coords, axis=0))
                pca = PCA(n_components=3)
                pca.fit(coords)
                u = -np.flip(pca.components_)[0]

                # Now decide it directed towards pre syn region
                pre = patch_y == 1
                coords = np.array(np.where(pre)).transpose()
                pre_center = np.flip(np.mean(coords, axis=0))

                w = pre_center - syn_center
                angle = np.arccos(np.dot(u, w)/norm(u)/norm(w)) * 180/np.pi
                if angle > 90:
                    u = -u

                orientation = u
                scale = 1

                samples.append(Sample(patch_x, patch_y, orientation, centre, scale))

            data[data_mode] = CortexVoxelDataset(samples, cfg, data_mode)


        with open(data_root + 'labels/pre_computed.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



        return data


    def evaluate(self, target, pred, cfg):

        if cfg.model_name == 'panet':
            target = target[0]

        val = jaccard_index(target, pred, cfg.num_classes)
        return val


    def update_checkpoint(self, best_so_far, new_value):
        new_value = new_value[DataModes.VALIDATION]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.VALIDATION]
            return True if np.mean(new_value) > np.mean(best_so_far) else False




