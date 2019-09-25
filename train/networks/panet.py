from itertools import chain
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_common import crop, crop_and_merge
from utils import transforms, affine_3d_grid_generator
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

        feature_channel_count = config.config.pose_latent_feature_channel_count * 2 ** config.steps
        self.fc_loc_laye1_size = feature_channel_count * config.config.pose_latent_feature_dim

        self.localization = lambda x: x[:, :feature_channel_count]


        self.fc_loc = nn.Sequential(
                        nn.Linear(self.fc_loc_laye1_size, self.fc_loc_laye1_size // 16),        nn.ReLU(True),
                        nn.Linear(self.fc_loc_laye1_size // 16, self.fc_loc_laye1_size // 64),  nn.ReLU(True),
                        nn.Linear(self.fc_loc_laye1_size // 64, config.config.theta_param_count))

        self.prior = config.config.priors.cuda()
        self.upsample_mode = 'trilinear' if config.ndims == 3 else 'bilinear'
        self.center = tuple([i // 2 for i in self.prior.shape[2:]])


    # estimate pose
    def find_pose(self, x_list):

        # estimate parameters
        xs = self.localization(x_list[-1])
        xs = xs.view(xs.shape[0], -1)
        params = self.fc_loc(xs)

        return params


    def register_atlas(self, atlas, pose, size_out):


        direction = torch.Tensor([0, -1, 0]).cuda()[None]
        q = direction + pose[:, :3]
        q = F.normalize(q, dim=1)
        theta_q = transforms.rot_matrix_from_quaternion(q)


        theta_shift = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1).cuda().float()
        theta_shift[:, :3, 3] = pose[:, 3:6]


        # theta_scale = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1).float() * axis[:, 6]
        # theta_scale[:, 3, 3] = 1
        # theta = theta_scale @ theta_q @ theta_shift

        theta = theta_q @ theta_shift

        theta = theta[:, :3]
        grid = affine_3d_grid_generator.affine_grid(theta, atlas.size()).double()
        rotated = F.grid_sample(atlas.double(), grid, mode='bilinear', padding_mode='zeros').float().detach()

        _, _, D_out, H_out, W_out = size_out
        N_in, C_in, D_in, H_in, W_in = rotated.shape
        center = (D_in//2, H_in//2, W_in//2)
        rotated = crop(rotated, (N_in, C_in, D_out, H_out, W_out), (0, self.config.config.prior_channel_count//2,) + center)

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
        prior = self.prior.repeat(*([input.shape[0], 1] + [1 for _ in range(self.config.ndims)]))
        pose = self.find_pose(down_outputs)
        registered_atlas = self.register_atlas(prior , pose, input.shape)

        # Uncomment if the priors are registered based on pose of each structure separately
        # In pose first n parameters corresponds to the pose of structure 1 and last n parameters corresponds to pose of structure 2
        # registered_atlas_1 = self.register_atlas(self.prior[:, 0], axis[:, :n])
        # registered_atlas_2 = self.register_atlas(self.prior[:, 1], axis[:, n:])
        # registered_atlas = torch.cat((registered_atlas_1, registered_atlas_2), dim=1)

        # up layers
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = crop_and_merge(F.upsample(registered_atlas, size=x.shape[2:], mode=self.upsample_mode), x) # PAs
            x = unet_layer(x)

        x = crop_and_merge(F.upsample(registered_atlas, size=x.shape[2:], mode=self.upsample_mode), x) # PAs
        x = self.final_layer(x)

        return x, pose


    def loss(self, input, target):

        y_seg = target[0]
        y_orient = target[1]

        y_seg_hat, y_orient_hat = self.forward(input)

        CE_Loss = nn.CrossEntropyLoss()
        MSE_Loss = nn.MSELoss()

        ce_loss = CE_Loss(y_seg_hat,  y_seg)
        mse_loss = MSE_Loss(y_orient_hat, y_orient)

        loss = self.config.config.lamda_angle * mse_loss + self.config.config.lamda_ce * ce_loss

        log = {"loss": loss.detach(), "ce_loss": ce_loss.detach(), "angle_loss": mse_loss.detach()}

        return loss, log
