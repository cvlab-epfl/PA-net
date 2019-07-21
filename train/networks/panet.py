from itertools import chain
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_common import crop, crop_and_merge
from utils import stns, affine_3d_grid_generator
import torch


class UNetLayer(nn.Module):

    def __init__(self, num_channels_in, num_channels_out, ndims):

        super(UNetLayer, self).__init__()

        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        conv1 = conv_op(num_channels_in,  num_channels_out, kernel_size=3, padding=1)
        conv2 = conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1)

        # batch_nrom_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d
        # bn1 = batch_nrom_op(num_channels_out)
        # bn2 = batch_nrom_op(num_channels_out)

        self.unet_layer = nn.Sequential(conv1, nn.ReLU() , conv2, nn.ReLU())
        # self.unet_layer = nn.Sequential(conv1, bn1, relu1, conv2, bn2, relu2)

    def forward(self, x):
        return self.unet_layer(x)


class PANet(nn.Module):
    """The U-Net."""

    def __init__(self, config):

        super(PANet, self).__init__()
        self.config = config
        assert config.ndims == 2 or config.ndims ==3, Exception("Invalid nidm: {}".format(config.ndims))

        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2)
        ConvLayer = nn.Conv3d if config.ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if config.ndims == 3 else nn.MaxPool2d(2)

        prior_channels = config.config.priors.shape[1] 

        '''  Down layers '''
        down_layers = [UNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            lyr = UNetLayer(config.first_layer_channels * 2**(i - 1), config.first_layer_channels * 2**i, config.ndims)
            down_layers.append(lyr)

        ''' Up layers '''
        up_layers = []
        for i in range(config.steps - 1, -1, -1):
            upconv = ConvTransposeLayer(in_channels=config.first_layer_channels   * 2**(i+1), out_channels=config.first_layer_channels * 2**i, kernel_size=2, stride=2)
            lyr = UNetLayer(config.first_layer_channels * 2**(i + 1) + prior_channels, config.first_layer_channels * 2**i, config.ndims)
            up_layers.append((upconv, lyr))

        ''' Final layer '''
        final_layer = ConvLayer(in_channels=config.first_layer_channels + prior_channels, out_channels=config.num_classes, kernel_size=1)

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers))
        self.final_layer = final_layer



        feature_channel_count = 8 * 2 ** config.steps
        self.fc_loc_laye1_size = feature_channel_count * 6 * 6 * 6
        # self.fc_loc_laye1_size = feature_channel_count * 18 * 14 * 1
        self.localization = lambda x: x[:, :feature_channel_count]


        self.fc_loc = nn.Sequential(
                        nn.Linear(self.fc_loc_laye1_size, self.fc_loc_laye1_size // 16),        nn.ReLU(True),
                        nn.Linear(self.fc_loc_laye1_size // 16, self.fc_loc_laye1_size // 64),  nn.ReLU(True),
                        nn.Linear(self.fc_loc_laye1_size // 64, config.config.theta_param_count))

        self.prior = config.config.priors.repeat(*([config.batch_size, 1] + [1 for _ in range(config.ndims)])).cuda()
        self.upsample_mode = 'trilinear' if config.ndims == 3 else 'bilinear'
        self.center = tuple([i // 2 for i in self.prior.shape[2:]])


    # estimate pose
    def find_pose(self, x_list):

        # estimate parameters
        xs = self.localization(x_list[-1])
        xs = xs.view(-1, self.fc_loc_laye1_size)
        params = self.fc_loc(xs)

        return params


    def register_atlas(self, x, axis, size_out):


        theta_shift = torch.eye(4, 4)[None].repeat(axis.shape[0], 1, 1).cuda().float()
        theta_shift[:, 0, 3] = axis[:, 3]
        theta_shift[:, 1, 3] = axis[:, 4]
        theta_shift[:, 2, 3] = axis[:, 5]

        direction = torch.Tensor([0, -1, 0, 0]).cuda()[:3][None]
        q = direction + axis[:, :3]
        q = F.normalize(q, dim=1)

        theta_q = stns.stn_batch_quaternion_rotations(q)

        # theta_scale = torch.eye(4, 4)[None].repeat(x.shape[0], 1, 1).cuda().float() * axis[:, 6] / 0.125
        # theta_scale[:, 3, 3] = 1
        # theta = theta_scale @ theta_q @ theta_shift

        theta = theta_q @ theta_shift

        theta = theta[:, 0:3, :]
        grid = affine_3d_grid_generator.affine_grid(theta, x.size()).double()
        rotated = F.grid_sample(x.double(), grid, mode='bilinear', padding_mode='zeros').float().detach()

        _, _, D_out, H_out, W_out = size_out
        N, C, D, H, W = rotated.shape
        center = (D//2, H//2, W//2)
        rotated = crop(rotated, (N, C, D_out, H_out, W_out) + (), (0, 2,) + center)
        # crop_shape = tuple([N, x.shape[1], D_out, H_out, W_out])
        # to_crop = tuple([False, False, True, True, True])
        # center = tuple([s // 2 for s in rotated.shape])
        #
        # rotated = patch_utils.get_patch(rotated, crop_shape, center, mode='constant', copy=False, to_crop=to_crop).patch
        # rotated = rotated.detach()

        return rotated

    def forward(self, input):

        # input
        x = input

        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)

        # pose estimation and atlas registration
        pose = self.find_pose(down_outputs)
        registered_atlas = self.register_atlas(self.prior, pose, input.shape)

        # up layers
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = crop_and_merge(F.upsample(registered_atlas, size=x.shape[2:], mode=self.upsample_mode), x) # mask
            x = unet_layer(x)

        x = crop_and_merge(F.upsample(registered_atlas, size=x.shape[2:], mode=self.upsample_mode), x) # mask
        x = self.final_layer(x)

        return x, pose


    def loss(self, x, y, weighsts):

        y_seg = y[0]
        y_orient = y[1]

        y_seg_hat, y_orient_hat = self.forward(x)

        CE_Loss = nn.CrossEntropyLoss()
        MSE_Loss =   nn.MSELoss()

        ce_loss = CE_Loss(y_seg_hat,  y_seg)
        mse_loss =   MSE_Loss(y_orient_hat, y_orient)

        loss = self.config.config.lamda_angle * mse_loss + self.config.config.lamda_ce * ce_loss

        log = {"loss": loss.detach(), "ce_loss": ce_loss.detach(), "angle_loss": mse_loss.detach()}

        return loss, log
