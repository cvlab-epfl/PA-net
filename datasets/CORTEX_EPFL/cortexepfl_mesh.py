import numpy as np
from skimage import io
from datasets.datasetandsupport import DatasetAndSupport

from evaluate.standard_metrics import jaccard_index
from utils.utils_common import invfreq_lossweights, crop, DataModes, crop_indices, blend
from utils.utils_mesh import run_rasterize, read_obj

from utils import stns
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
import itertools as itr
import torch

class Sample:
    def __init__(self, x, y, orientation, w, center):
        self.x = x
        self.y = y
        self.w = w
        self.orientation = orientation
        self.center = center

class Part:

    def __init__(self, part_id, vertices, faces, centroid, scale, N):
        self.part_id = part_id
        self.vertices = vertices
        self.faces = faces
        self.centroid = centroid
        self.scale = scale
        self.N = N

class CortexMeshDataset(Dataset):

    def __init__(self, data, cfg, mode):
        self.data = data
        self.cfg = cfg
        self.mode = mode

        self.sphere_vertices, self.sphere_faces = read_obj(cfg.sphere_path)
        self.N = len(self.sphere_vertices)

        self.sphere_vertices = torch.from_numpy(self.sphere_vertices).cuda().float()
        # self.sphere_faces = torch.from_numpy(self.sphere_faces).cuda()

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
            new_orientation = torch.rand(3) - 0.5
            new_orientation = F.normalize(new_orientation, dim=0)
            q = orientation + new_orientation
            q = F.normalize(q, dim=0)
            theta_rotate = stns.stn_quaternion_rotations(q, on_gpu=False)

            shift = 2 * (torch.rand(3) - 0.5) * self.cfg.augmentation_shift_range
            normalized_shift = torch.tensor([d/(D//2) for d, D in zip(shift, y.shape)])
            theta_shift = stns.shift(normalized_shift)
            theta = theta_rotate @ theta_shift

            x, y, w = stns.transform(theta, x, y)
            orientation = new_orientation

        else:
            shift = torch.zeros(3).float()

        pose_structure = torch.cat((orientation, shift))

        C, D, H, W = x.shape
        center = (D//2, H//2, W//2)

        x = crop(x, (C,) + self.cfg.patch_shape, (0,) + center)
        y_voxels = crop(y, self.cfg.patch_shape, center)

        y_meshes = []
        C, D, H, W = x.shape # new shape

        # compute theta
        new_orientation = torch.tensor([0, -1, 0]).float()
        new_orientation = F.normalize(new_orientation, dim=0)
        q = orientation + new_orientation
        q = F.normalize(q, dim=0)
        theta_rotate = stns.stn_quaternion_rotations(q, on_gpu=False).cuda()

        pre_theta_shift = stns.shift(shift - torch.tensor([96//2, 96//2, 96//2]).float()).cuda()

        adict = {}
        for i in range(1, 4):

            y = (y_voxels == i).long()
            surface = F.max_pool3d(y[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0].long() - y
            surface = surface.data.cpu().numpy()

            surface_points = np.array(np.where(surface)).transpose()
            surface_points = np.flip(surface_points, axis=1) # convert z,y,x -> x, y, z
            surface_points = torch.from_numpy(surface_points.copy()).cuda().float()
            surface_centroid = torch.mean(surface_points, dim=0)[None]
            surface_points = surface_points - surface_centroid  # np.array([[D//2, H//2, W//2]])

            t = self.sphere_vertices @ surface_points.transpose(1, 0)

            projections = t.transpose(1, 0)[:, :, None] * self.sphere_vertices

            distance = torch.sqrt(torch.sum((projections - surface_points[:, None]) ** 2, dim=2)).transpose(1, 0)
            distance[t < 0] = 1000

            indices = torch.argmin(distance, dim=1)
            indices = indices[:, None].repeat(1, 3)[None]
            y_points = torch.gather(projections, 0, indices)[0]
            y_points = y_points + surface_centroid

            for dim, margin in enumerate([D, H, W]):
                y_points[y_points[:, dim] > margin, dim] = margin
                y_points[y_points[:, dim] < 0, dim] = 0

            # un pose
            temp = torch.cat([y_points, torch.ones((self.N, 1)).cuda()], dim=1).transpose(1, 0)
            rectified = theta_rotate @ pre_theta_shift @ temp # structure centriod shift and re-orientation
            y_points = rectified[:3].transpose(1, 0)

            adict['var_' + str(i)] = y_points.data.cpu().numpy()

            # compute centroid (new)
            centroid = torch.mean(y_points, dim=0)[None] # part centroid shift
            y_points = y_points - centroid

            # compute scale
            scale = torch.sqrt(torch.sum(y_points ** 2)) # part re-scaling
            y_points = y_points / scale


            y_points = y_points.contiguous().view(1, -1)[0]

            part = {}
            part[i] = i
            part['vertices'] = y_points
            part['faces'] = self.sphere_faces
            part['centroid'] = centroid
            part['scale'] = scale
            part['N'] = self.N
            part['parent_pose'] = pose_structure
            y_meshes.append(part)


        orientation = pose_structure[:3]
        shift = pose_structure[3:]
        post_theta_shift = stns.shift(-(shift - torch.tensor([96 // 2, 96 // 2, 96 // 2]).float())).cuda()

        new_orientation = torch.tensor([0, -1, 0]).float()
        new_orientation = F.normalize(new_orientation, dim=0)
        q = orientation + new_orientation
        q = F.normalize(q, dim=0)
        theta_rotate = stns.stn_quaternion_rotations(q, on_gpu=False).cuda()

        y_voxels_reconstructed = torch.zeros_like(y_voxels)
        for i, part in enumerate(y_meshes):

            vertices = part['vertices'].reshape(part['N'], 3) * part['scale'] + part['centroid']

            temp = torch.cat([vertices, torch.ones((self.N, 1)).cuda()], dim=1).transpose(1, 0)
            rectified = post_theta_shift @ theta_rotate @ temp
            vertices = rectified[:3].transpose(1, 0)
            vertices = torch.flip(vertices, [1])

            vertices = vertices.data.cpu().numpy()

            faces_ = part['faces']
            y_voxels_i = run_rasterize(vertices, faces_, grid_size=y_voxels.shape)
            y_voxels_i = torch.from_numpy(y_voxels_i).cuda() / 255

            y_voxels_reconstructed[y_voxels_i == 1] = i+1

        # adict['orientation'] = orientation.data.cpu().numpy()
        # from scipy.io import savemat
        # savemat('/cvlabdata1/cvlab/datasets_udaranga/mesh_' + str(idx), adict)
        return x, y_voxels_reconstructed, y_meshes


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
        counts = [[0, 4, 6, 8, 10, 11, 12, 16, 17, 19, 20, 21, 22, 24], range(24, 36), range(24, 36)]

        data = {}

        patch_shape_extended = tuple([int(np.sqrt(2) * i) + 2 * cfg.augmentation_shift_range for i in cfg.patch_shape])

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]):

            samples = []
            for j in counts[i]:

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
                coords = np.flip(coords, axis=1) # make it x,y,z
                syn_center = np.mean(coords, axis=0)
                pca = PCA(n_components=3)
                pca.fit(coords)
                # u = -np.flip(pca.components_)[0]
                u = pca.components_[2]

                # Now decide it directed towards pre syn region
                pre = patch_y == 1
                coords = np.array(np.where(pre)).transpose()
                coords = np.flip(coords, axis=1)  # make it x,y,z
                # pre_center = np.flip(np.mean(coords, axis=0))
                pre_center = np.mean(coords, axis=0)

                w = pre_center - syn_center
                angle = np.arccos(np.dot(u,w)/norm(u)/norm(w)) * 180/np.pi
                if angle > 90:
                    u = -u

                orientation = u

                # compare with mannual ground truth
                # orientation = idx[j][3:6]
                # angle = np.arccos(np.dot(u,v)/norm(u)/norm(v)) * 180/np.pi
                # print(angle)
                samples.append(Sample(patch_x, patch_y, orientation, patch_w, centre))

            data[datamode] = CortexMeshDataset(samples, cfg, datamode)


        with open(data_root + data_version + 'labels/pre_computed.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

        return data


    def evaluate(self, target, pred, cfg):
        # target_seg = target[0] panet
        # val = torch.mean(torch.sqrt((target-pred) ** 2))
        # float(val.data.cpu().numpy())
        target_seg = target
        val = jaccard_index(target_seg, pred, cfg.num_classes)
        return val


    def update_checkpoint(self, best_so_far, new_value):
        new_value = new_value[DataModes.VALIDATION]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.VALIDATION]
            return True if np.mean(new_value) > np.mean(best_so_far) else False




# from scipy.io import savemat
# item = self.data[idx]
# x = torch.from_numpy(item.x).cuda()[None]
# y = torch.from_numpy(item.y).cuda().long()
# C, D, H, W = x.shape
# center = (D // 2, H // 2, W // 2)
# # center = tuple(np.uint16(np.mean(np.where(y.data.cpu().numpy()==1),axis=1)))
# x = crop(x, (C,) + self.cfg.patch_shape, (0,) + center)
# y_voxels = crop(y, self.cfg.patch_shape, center)
# # w = crop(w, self.cfg.patch_shape, center)==1
# y_meshes = []
# C, D, H, W = x.shape  # new shape
# y = (y_voxels == 2).long()
# surface = F.max_pool3d(y[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0].long() - y
# surface = surface.data.cpu().numpy()
# surface_points = np.array(np.where(surface)).transpose()
# surface_points = torch.from_numpy(surface_points).cuda().float()
# surface_centroid = torch.mean(surface_points, dim=0)[None]
# surface_points = surface_points - surface_centroid  # np.array([[D//2, H//2, W//2]])
# t = self.sphere_vertices @ surface_points.transpose(1, 0)
# projections = t.transpose(1, 0)[:, :, None] * self.sphere_vertices
# distance = torch.sqrt(torch.sum((projections - surface_points[:, None]) ** 2, dim=2)).transpose(1, 0)
# distance[t < 0] = 1000
# indices = torch.argmin(distance, dim=1)
# indices = indices[:, None].repeat(1, 3)[None]
# y_points = torch.gather(projections, 0, indices)[0]
# y_points = y_points + surface_centroid
# coords = y_points.data.cpu().numpy()
# syn_center = np.flip(np.mean(coords, axis=0))
# pca = PCA(n_components=3)
# pca.fit(coords)
# u = -np.flip(pca.components_)[0]
# orientation = torch.from_numpy(u).float()
# new_orientation = torch.tensor([0, -1, 0]).float()
# new_orientation = F.normalize(new_orientation, dim=0)
# q = orientation + new_orientation
# q = F.normalize(q, dim=0)
# theta = stns.stn_quaternion_rotations(q, on_gpu=False)
# fig = pyplot.figure()
# ax = Axes3D(fig)
# adict = {}
# for i in range(1, 4):
#     y = (y_voxels == i).long()
#     surface = F.max_pool3d(y[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0].long() - y
#     surface = surface.data.cpu().numpy()
#     surface_points = np.array(np.where(surface)).transpose()
#     surface_points = torch.from_numpy(surface_points).cuda().float()
#     surface_centroid = torch.mean(surface_points, dim=0)[None]
#     surface_points = surface_points - surface_centroid  # np.array([[D//2, H//2, W//2]])
#     t = self.sphere_vertices @ surface_points.transpose(1, 0)
#     projections = t.transpose(1, 0)[:, :, None] * self.sphere_vertices
#     distance = torch.sqrt(torch.sum((projections - surface_points[:, None]) ** 2, dim=2)).transpose(1, 0)
#     distance[t < 0] = 1000
#     indices = torch.argmin(distance, dim=1)
#     indices = indices[:, None].repeat(1, 3)[None]
#     y_points = torch.gather(projections, 0, indices)[0]
#     y_points = y_points + surface_centroid
#     for dim, margin in enumerate([D, H, W]):
#         y_points[y_points[:, dim] > margin, dim] = margin
#         y_points[y_points[:, dim] < 0, dim] = 0
#     # un pose
#     y_points = y_points - 96 // 2
#     temp = torch.cat([y_points, torch.ones((self.N, 1)).cuda()], dim=1)
#     rectified = temp @ theta.cuda()
#     a = rectified[:, :3]
#     # ax.scatter(a[:, 0].data.cpu().numpy(), a[:, 1].data.cpu().numpy(), a[:, 2].data.cpu().numpy())
#     adict['var_'+str(i)] = a.data.cpu().numpy()
# savemat('/cvlabdata1/cvlab/datasets_udaranga/mesh',adict)




        # y = (y_voxels == 2).long()
        # surface = F.max_pool3d(y[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0].long() - y
        # surface = surface.data.cpu().numpy()
        #
        # surface_points = np.array(np.where(surface)).transpose()
        # surface_points = torch.from_numpy(surface_points).cuda().float()
        # surface_centroid = torch.mean(surface_points, dim=0)[None]
        # surface_points = surface_points - surface_centroid  # np.array([[D//2, H//2, W//2]])
        #
        # t = self.sphere_vertices @ surface_points.transpose(1, 0)
        #
        # projections = t.transpose(1, 0)[:, :, None] * self.sphere_vertices
        #
        # distance = torch.sqrt(torch.sum((projections - surface_points[:, None]) ** 2, dim=2)).transpose(1, 0)
        # distance[t < 0] = 1000
        #
        # indices = torch.argmin(distance, dim=1)
        # indices = indices[:, None].repeat(1, 3)[None]
        # y_points = torch.gather(projections, 0, indices)[0]
        # y_points = y_points + surface_centroid
        #
        # for dim, margin in enumerate([D, H, W]):
        #     y_points[y_points[:, dim] > margin, dim] = margin
        #     y_points[y_points[:, dim] < 0, dim] = 0
        #
        # coords = y_points.data.cpu().numpy() - 96//2
        # syn_center = np.flip(np.mean(coords, axis=0))
        # pca = PCA(n_components=3)
        # pca.fit(coords)
        # # orientation = -np.flip(pca.components_)[0]
        # orientation = (pca.components_)[2]
        # orientation_old = item.orientation
        # print(np.arccos(np.dot(orientation, orientation_old))*180/np.pi)
        #
        # orientation = torch.tensor(orientation).float()
