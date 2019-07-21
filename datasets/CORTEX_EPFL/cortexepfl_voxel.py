import numpy as np
from skimage import io
from datasets.datasetandsupport import DatasetAndSupport

from evaluate.standard_metrics import jaccard_index
from utils.utils_common import invfreq_lossweights, crop, DataModes, crop_indices
from utils import stns
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
import itertools as itr

class Sample:
    def __init__(self, x, y, orientation, w, center):
        self.x = x
        self.y = y
        self.w = w
        self.orientation = orientation
        self.center = center



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
        if self.mode == 'all':
            orientation = None
        else:
            orientation = torch.from_numpy(item.orientation).float()


        if self.mode == DataModes.TRAINING: # if training do augmentation
            new_orientation = (torch.rand(3) - 0.5) * 2 * self.cfg.augmentation_shift_range
            new_orientation = F.normalize(new_orientation, dim=0)
            q = orientation + new_orientation
            q = F.normalize(q, dim=0)
            theta_rotate = stns. stn_quaternion_rotations(q, on_gpu=False)

            shift = torch.tensor([d/(D//2) for d, D in zip(2 * (torch.rand(3) - 0.5) * self.cfg.augmentation_shift_range, y.shape)])
            theta_shift = stns.shift(shift)
            theta =  theta_rotate @ theta_shift


            # w = torch.zeros_like(y).cuda().float()
            # w[195 // 2] = 1
            # w[:, 195 // 2] = 1
            # w[:, :, 195 // 2] = 1
            #
            # x, y, w = stns.transform(theta, x, y * w, w)
            x, y, w = stns.transform(theta, x, y)
            orientation = new_orientation

            pose = torch.cat((orientation, shift)).cuda()
        else:
            pose = torch.zeros(6).cuda()
            w = torch.zeros_like(y)


        C, D, H, W = x.shape
        center = (D//2, H//2, W//2)
        if self.mode == 'all':
            x_in = x
            y_in = y
            x = [crop(x_in, (C,) + self.cfg.patch_shape, (0,) + center)]
            y = [crop(y_in, self.cfg.patch_shape, center)]
            stack_loc = [item.center]

            center_shifts = list(itr.product([-20, 20], repeat=3))
            for shift in center_shifts:
                center_i = tuple([ c + s for s,c in zip(shift, center)])
                x.append(crop(x_in, (C,) + self.cfg.patch_shape, (0,) + center_i))
                y.append(crop(y_in, self.cfg.patch_shape, center_i))
                stack_loc.append(tuple([ c + s for s, c in zip(shift, item.center)]))

            w = stack_loc
        else:

            x = crop(x, (C,) + self.cfg.patch_shape, (0,) + center)
            y = crop(y, self.cfg.patch_shape, center)
            # w = crop(w, self.cfg.patch_shape, center)==1
            w = item.center # dummy value since it's not used
        # return x, [y, pose], w
        return x, y, w


class CortexEpfl(DatasetAndSupport):

    def __init__(self):
        self.x = None
        self.y = None


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

        data_root = '/cvlabdata1/cvlab/datasets_udaranga/datasets/3d/graham/'
        class_id = 14
        num_classes = 4
        data_version = 'labels_v' + str(class_id) + '/'
        path_images = data_root + 'imagestack_downscaled.tif'
        path_synapse = data_root + data_version + 'labels/labels_synapses_' + str(class_id) + '.tif'
        path_pre_post = data_root + data_version + 'labels/labels_pre_post_' + str(class_id) + '.tif'

        ''' Label information '''
        # path_idx = data_root + data_version + 'labels/info.txt'
        # idx = np.loadtxt(path_idx)

        path_idx = data_root + data_version + 'labels/info.npy'
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
        # counts = [[0, 12], [12, 24], [24, 36]]
        # counts = [[0, 1], [12, 13], [24, 25]]
        counts = [[0, 1, 2, 4, 6, 7, 8, 10, 11, 12, 15, 16], range(17, 24), range(24, 36)]

        data = {}

        patch_shape_extended = tuple([int(np.sqrt(2) * i) + 2 * cfg.augmentation_shift_range for i in cfg.patch_shape])

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
                patch_w = invfreq_lossweights(patch_y, num_classes)

                # Compute orientation

                # First find the Axis
                syn = patch_y == 2
                coords = np.array(np.where(syn)).transpose()
                syn_center = np.flip(np.mean(coords,axis=0))
                pca = PCA(n_components=3)
                pca.fit(coords)
                u = -np.flip(pca.components_)[0]

                # Now decide it directed towards pre syn region
                pre = patch_y == 1
                coords = np.array(np.where(pre)).transpose()
                pre_center = np.flip(np.mean(coords,axis=0))

                w = pre_center - syn_center
                angle = np.arccos(np.dot(u,w)/norm(u)/norm(w)) * 180/np.pi
                if angle > 90:
                    u = -u

                orientation = u

                # compare with mannual ground truth
                # v = idx[j][3:6]
                # angle = np.arccos(np.dot(u,v)/norm(u)/norm(v)) * 180/np.pi
                # print(angle)

                samples.append(Sample(patch_x, patch_y, orientation, patch_w, centre))

            data[data_mode] = CortexVoxelDataset(samples, cfg, data_mode)


        with open(data_root + data_version + 'labels/pre_computed.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

        return data

    def load_data_all_in_one(self, cfg):
        '''
        # Change this to load your training data.

        # pre-synaptic neuron   :   1
        # synapse               :   2
        # post-synaptic neuron  :   3
        # background            :   0
        '''

        data_root = '/cvlabdata1/cvlab/datasets_udaranga/datasets/3d/graham/'
        class_id = 14
        num_classes = 4
        data_version = 'labels_v' + str(class_id) + '/'
        path_images = data_root + 'imagestack_downscaled.tif'

        path_pre_post = data_root + data_version + 'labels/labels_pre_post_' + str(class_id) + '.tif'
        seeds = data_root + data_version + 'labels/seeds.tif'

        ''' Label information '''
        # path_idx = data_root + data_version + 'labels/info.txt'
        # idx = np.loadtxt(path_idx)

        path_idx = data_root + data_version + 'labels/info.npy'
        idx = np.load(path_idx)

        ''' Load data '''
        x = io.imread(path_images)
        y_seeds = io.imread(seeds)
        temp = io.imread(path_pre_post)
        y_pre_post = np.zeros_like(y_seeds)
        y_pre_post[:200] = temp

        x = np.float32(x) / 255
        y_synapses = np.int64(y_seeds)

        # Syn at bottom
        temp = np.int64(y_pre_post)
        y_pre_post = np.copy(y_synapses)
        y_pre_post[temp > 0] = temp[temp > 0]

        # method 1: split neurons
        patch_shape_extended = tuple([int(np.sqrt(2) * i) for i in cfg.patch_shape])

        samples = []

        ids = np.unique(y_synapses)
        ids = np.delete(ids, 0)
        for j in range(len(ids)):

            points = np.where(y_synapses == ids[j])
            centre = tuple(np.mean(points, axis=1, dtype=np.int64))

            # extract the object of intereset
            y = np.zeros_like(y_pre_post)

            loc_in_idx = np.where(idx[:, 1] == ids[j])[0]
            if loc_in_idx.size > 0:
                for k, id in enumerate(idx[loc_in_idx[0]][:3]):
                    y[y_pre_post == id] = k + 1

            patch_y = crop(y, patch_shape_extended, centre)
            patch_x = crop(x, patch_shape_extended, centre)
            patch_w = invfreq_lossweights(patch_y, num_classes)

            orientation = None

            samples.append(Sample(patch_x, patch_y, orientation, patch_w, centre))

        data = CortexVoxelDataset(samples, cfg, 'all')

        with open(data_root + data_version + 'labels/pre_computed_all.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(data_root + data_version + 'labels/pre_computed_all.pickle', 'rb') as handle:
        #     data = pickle.load(handle)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y_pre_post)

        return data

    def evaluate(self, target, pred, cfg):
        # target_seg = target[0] panet
        val = jaccard_index(target, pred, cfg.num_classes)
        return val


    def update_checkpoint(self, best_so_far, new_value):
        new_value = new_value[DataModes.VALIDATION]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.VALIDATION]
            return True if np.mean(new_value) > np.mean(best_so_far) else False




