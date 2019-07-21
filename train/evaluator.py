from utils.utils_common import DataModes, makedirs, append_line, write_lines, blend, crop_indices, blend_cpu
from utils.utils_mesh import run_rasterize
from torch.utils.data import DataLoader
import numpy as np
import torch
from skimage import io
from evaluate.standard_metrics import jaccard_index
import itertools
import torch.nn.functional as F

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

        if name == 'unet':
            x, y, w = data
            y_hat = self.net(x)
            y_hat = torch.argmax(y_hat, dim=1).cpu()
        elif name == 'panet':
            x, y, w = data
            y_hat = self.net(x)[0]  # panet
        elif name == 'conv_vae':
            x, y, w = data
            input_ = torch.zeros_like(y[None]).repeat(1, 4, 1, 1, 1)
            input_ = input_.scatter_(1, y[None], torch.ones_like(y[None])).float()
            y_hat, _, _ = self.net(input_)
        elif name == 'mesh_in_mesh_out':
            x, y, w = data
            y_hat = self.net(y)

            if len(y_hat) == 3: # true when vae is used
                y_hat = y_hat[0]

            centroid = w[2]
            scale = w[3]

            D, H, W = x.shape[2:]


            # y_ = y.view(1, 642, 3)*scale
            # y_hat_ = y_hat.view(1, 642, 3)*scale
            # error_mean = torch.mean(torch.sqrt(torch.sum((y_ - y_hat_) ** 2, dim=2))).data.cpu().numpy()
            # error_max = torch.max(torch.sqrt(torch.sum((y_ - y_hat_) ** 2, dim=2))).data.cpu().numpy()
            # print('mean: ' + str(error_mean) + ' max:' + str(error_max))


            vertices = y_hat.reshape(-1, self.net.N, 3)*scale + centroid + torch.tensor([[[D//2, H//2, W//2]]]).cuda().float()
            vertices = vertices.data.cpu().numpy()[0]

            faces_  = w[1][0].data.cpu().numpy()
            y_hat = run_rasterize(vertices, faces_, grid_size=tuple([D, H, W]))[None]
            y_hat = torch.from_numpy(y_hat).cuda()/255

            y = w[0].cpu()

        elif name == 'pca':
            x, y, w = data
            y_hat = self.net(y)
            y_hat = y_hat.data.cpu().numpy()

            grid_size = x.shape[2:]
            vertices = np.array(grid_size)[None] * (y_hat.reshape(-1, self.net.N, 3)[0] + 1)/2
            faces_  = w[1][0].data.cpu().numpy()
            y_hat = run_rasterize(vertices, faces_, grid_size=grid_size)[None]
            y_hat = torch.from_numpy(y_hat).cuda()/255

            y = w[0].cpu()

        elif name == 'voxel_in_mesh_out':
            from utils import stns

            x, target_voxels, target_mesh = data

            # input = w[0][None].float()
            pred_meshes, pred_vox = self.net(x)

            pose_structure = target_mesh[0]['parent_pose'] # parent pose is stored inside parts (and is same in all parts)
            orientation = pose_structure[0, :3]
            shift = pose_structure[0, 3:]
            post_theta_shift = stns.shift(-(shift - torch.tensor([96 // 2, 96 // 2, 96 // 2]).float())).cuda()

            new_orientation = torch.tensor([0, -1, 0]).float()
            new_orientation = F.normalize(new_orientation, dim=0)
            q = orientation + new_orientation
            q = F.normalize(q, dim=0)
            theta_rotate = stns.stn_quaternion_rotations(q, on_gpu=False).cuda()

            pred_voxels_reconstructed = torch.zeros_like(target_voxels[0])
            for i, (target, pred) in enumerate(zip(target_mesh, pred_meshes)):
                part, part_pose = pred
                vertices = part.reshape(target['N'][0], 3) * target['scale'] + target['centroid'][0]

                temp = torch.cat([vertices, torch.ones((target['N'], 1)).cuda()], dim=1).transpose(1, 0)
                rectified = post_theta_shift @ theta_rotate @ temp
                vertices = rectified[:3].transpose(1, 0)
                vertices = torch.flip(vertices, [1])

                vertices = vertices.data.cpu().numpy()

                faces_ = target['faces'][0].data.cpu().numpy()
                y_voxels_i = run_rasterize(vertices, faces_, grid_size=target_voxels.shape[1:])
                y_voxels_i = torch.from_numpy(y_voxels_i).cuda() / 255

                pred_voxels_reconstructed[y_voxels_i == 1] = i + 1


            y =  target_voxels
            y_hat = pred_voxels_reconstructed[None]

            # y_hat = torch.argmax(pred_vox, dim=1).cpu()
            # y_hat = F.upsample(y_hat[:, None].float(), scale_factor=4, mode='nearest')[:, 0].long()


        return y, y_hat

    def evaluate_set(self, dataloader):
        results = []
        for data in dataloader:
            y, y_hat = self.predict(data, self.config.name)
            results.append(self.support.evaluate(y, y_hat, self.config))

        results = np.array(results)
        return results

    def save_results(self, dataloader, epoch, performence, save_path):

        xs = []
        ys = []
        y_hats = []

        for i, data in enumerate(dataloader):
            x, y, center = data

            y, y_hat = self.predict(data, self.config.name)

            #
            y_hat = y_hat[0]

            xs.append(x[0, 0])
            ys.append(y[0])
            # ys.append(y[0][0]) panet
            y_hats.append(y_hat)

        if performence is not None:
            performence_mean = 100*np.mean(performence, axis=0)
            summary = ('{}: ' + ', '.join(['{:.2f}' for _ in range(self.config.num_classes-1)])).format(epoch, *performence_mean)
            print(save_path + '- performence: ' + summary)
            append_line(save_path + 'summary.txt', summary)
            all_results = [('{}: ' + ', '.join(['{:.2f}' for _ in range(self.config.num_classes-1)])).format(*((i+1,) + tuple(vals))) for i, vals in enumerate(performence)]
            write_lines(save_path + 'all_results.txt', all_results)

        xs = torch.cat(xs,dim=0).cpu()
        ys = torch.cat(ys,dim=0).cpu()
        y_hats = torch.cat(y_hats,dim=0).cpu()


        overlay_y_hat = blend_cpu(xs, y_hats, self.config.num_classes)
        overlay_y = blend_cpu(xs, ys, self.config.num_classes)
        overlay = np.concatenate([overlay_y, overlay_y_hat], axis=2)
        io.imsave(save_path + 'overlay_y_hat.tif', overlay)


    def merge_patch_into_stack(self, stack_shape, patch_shape, patch_center_in_stack):
        stack_slices, pad, _ = crop_indices(stack_shape, patch_shape, patch_center_in_stack)
        patch_slices = tuple([slice(p[0], dim - p[1]) for p, dim in zip(np.array(pad), patch_shape)])

        return patch_slices, stack_slices

    def save_results_complete_set_mesh(self, dataloader, epoch, performence):

        for i, data in enumerate(dataloader):

            x, y, centers = data

            print(i)

    def save_results_complete_set(self, dataloader, epoch, performence):

        xs = []
        ys = []
        y_hats = []
        y_hats_full = torch.zeros_like(self.support.x)
        y_full = torch.zeros_like(self.support.x).cuda()

        flips = [[2], [2,3], [2,4], [2,3,4], [3], [3,4], [2,4], [4]]


        center_shifts = [(0,0,0)]  + list(itertools.product([-self.config.eval_center_shift, self.config.eval_center_shift], repeat=3))

        for i, data in enumerate(dataloader):
            print(i)
            xs, _, centers = data


            y_hats = []
            # can't process the as a batch due to GPU memory limitations
            # can't do on cpu because trilinear interpolation is too slow on cpu
            for x in xs:
                y_hat = self.net(x)[0].detach().cpu()
                for flip in flips:
                    y_hat += torch.flip(self.net(torch.flip(x, dims = flip))[0].detach().cpu(), dims = flip)

                y_hats.append(torch.argmax(y_hat, dim=1)[0])



            for y_hat, center in zip(y_hats, centers):
                patch_slices, stack_slices = self.merge_patch_into_stack(self.support.x.shape, y_hat.shape, center)
                temp = torch.zeros_like(y_hats_full)
                temp[stack_slices] = y_hat[patch_slices]
                y_hats_full[temp>0] = temp[temp>0]

                # temp = torch.zeros_like(y_full)
                # temp[stack_slices] = y[0][0][patch_slices]
                # y_full[temp>0] = temp[temp>0]

        if performence is not None:
            performence_mean = 100*np.mean(performence, axis=0)
            summary = ('{}: ' + ', '.join(['{:.2f}' for _ in range(self.config.num_classes-1)])).format(epoch, *performence_mean)
            append_line(self.save_path + '/summary.txt', summary)
            all_results = [('{}: ' + ', '.join(['{:.2f}' for _ in range(self.config.num_classes-1)])).format(*((i+1,) + tuple(vals))) for i, vals in enumerate(performence)]
            write_lines(self.save_path + '/all_results.txt', all_results)
        else:
            performence = jaccard_index(y_full[100:200], y_hats_full[100:200], self.config.num_classes)

        # xs = torch.cat(xs,dim=0).cpu()
        # # ys = torch.cat(ys,dim=0)
        # y_hats = torch.cat(y_hats,dim=0).cpu()


        # overlay = blend_cpu(xs, y_hats, self.config.num_classes)
        # io.imsave(self.save_path + '/overlay_y_hat.tif', overlay)

        overlay = blend_cpu(self.support.x, y_hats_full.long().cpu(), self.config.num_classes)
        io.imsave(self.save_path + '/overlay_y_hat_full_extended.tif', overlay)

        io.imsave(self.save_path + '/y_hat_pre.tif', np.uint8(y_hats_full==1))
        io.imsave(self.save_path + '/y_hat_syn.tif', np.uint8(y_hats_full == 2))
        io.imsave(self.save_path + '/y_hat_post.tif', np.uint8(y_hats_full == 3))

        # overlay = blend(self.support.x.cuda(), y_full.long(), self.config.num_classes)
        # io.imsave(self.save_path + '/overlay_y_full.tif', overlay)

    def vae_visualize(self, dataloader, epoch):

        data1 = next(iter(dataloader))
        data2 = next(iter(dataloader))

        _, y1, _ = data1
        _, y2, _ = data2

        input1 = torch.zeros_like(y1[None]).repeat(1, 4, 1, 1, 1)
        input1 = input1.scatter_(1, y1[None], torch.ones_like(y1[None])).float()

        input2 = torch.zeros_like(y2[None]).repeat(1, 4, 1, 1, 1)
        input2 = input2.scatter_(1, y2[None], torch.ones_like(y2[None])).float()

        self.net.encoder(input1)

