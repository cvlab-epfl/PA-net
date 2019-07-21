import logging
from torch.utils.tensorboard import SummaryWriter
from utils.utils_common import DataModes
import torch
logger = logging.getLogger(__name__)

class Trainer(object):

    def training_step(self, data, epoch):


        # Get the minibatch
        x, y_voxel, y_mesh = data


        self.optimizer.zero_grad()
        loss, log = self.net.loss(x, y_voxel, y_mesh, epoch)
        loss.backward()
        self.optimizer.step()

        return log

    def __init__(self, net, trainloader, optimizer, epoch_count, eval_every, save_path, evaluator, log_msg):

        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer

        self.numb_of_epochs = epoch_count
        self.eval_every = eval_every
        self.save_path = save_path
        self.evaluator = evaluator
        self.log_msg = log_msg


    def do_pca(self, stat_epoch):

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        ys = []
        for i in range(1000):
            print(i)
            for data in self.trainloader:
                _, _, y_meshes = data
                # ys.append(y_meshes['vertices'].data.cpu().numpy())
                a = []
                for i in y_meshes:
                    a.append(i['vertices'][None])
                ys.append(torch.cat(a, dim=1).cpu())

        ys = torch.cat(ys, dim=0)
        mean_shape = torch.mean(ys, dim=0).data.numpy()

        save_path = '/cvlabdata1/cvlab/datasets_udaranga/datasets/3d/mesh_templates/pca/642_mean_shape.npy'

        import numpy as np
        np.save(save_path, mean_shape)



        # vertices_test = np.array(ys)
        # vertices = np.load('/cvlabdata2/cvlab/datasets_udaranga/datasets/3d/graham/labels_v14/labels/training_meshes.npy')

        # vertices_mean = np.mean(vertices, axis=0)
        # vertices = (vertices - vertices_mean)/np.std(vertices, axis=0)
        # vertices[np.isnan(vertices)] = 0
        # vertices = StandardScaler().fit_transform(vertices)

        # pca = PCA()
        # pcs = pca.fit(vertices)
        #
        # axes = pcs.components_
        #
        # # new_shapes = np.mean(vertices, axis=0)
        # # new_shapes = np.repeat(new_shapes[None], vertices.shape[0], axis=0)
        # #
        # # for k in range(10):
        # #     bk = vertices @ axes[k]
        # #     new_shapes += bk[:, None] * axes[k][None]
        #
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # m = 10
        # samples = vertices_test[m].reshape(1800, 3)
        # fig = plt.figure(1)
        # ax = Axes3D(fig)
        # ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
        # for index, t in enumerate([1, 10, 100, 300]):
        #
        #     mean_shape = np.mean(vertices, axis=0)
        #     new_shapes = mean_shape
        #     for k in range(t):
        #         bk = np.dot(vertices_test[m] - mean_shape, axes[k])
        #         new_shapes += bk * axes[k]
        #     samples = new_shapes.reshape(1800, 3)
        #     fig = plt.figure(index + 2)
        #     ax = Axes3D(fig)
        #     ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])

    def train(self, start_epoch):
        logger.info("Start training...")
        writer = SummaryWriter(self.save_path)

        for epoch in range(start_epoch, start_epoch + self.numb_of_epochs):  # loop over the dataset multiple times
            running_loss = {}
            for data in self.trainloader:
                # training step
                loss = self.training_step(data, epoch)

                # print statistics
                for key, value in loss.items():
                    running_loss[key] = (running_loss[key] + value) if key in running_loss else 0

            if epoch % self.eval_every == self.eval_every-1:  # print every K epochs
                logger.info('epoch: {}, tr_loss: {:4f}'.format(epoch, running_loss['loss'] / self.eval_every))

                for key, value in running_loss.items():
                    writer.add_scalar(DataModes.TRAINING + '/' + key, value, epoch)
                self.evaluator.evaluate(epoch, writer)
                running_loss = 0.0

        logger.info("... end training!")


    # def eval(self, data):
    #     x, y, w, angle = data
    #     patch_o, axis_hat = self.net(x)
