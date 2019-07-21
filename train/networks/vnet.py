from itertools import chain
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_common import numpytorch, crop_and_merge
import itertools
from utils import stns


class VNetLayer(nn.Module):
    def __init__(self,num_channels_in, num_channels_out, ndims, layer_count=1, border_mode='same', batch_norm=False):
        super(VNetLayer, self).__init__()

        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        kernel_size = 5
        if border_mode == 'valid':
            padding = 0
        elif border_mode == 'same':
            padding = kernel_size//2
        else:
            raise ValueError("unknown border_mode `{}`".format(border_mode))


        layers = []
        for i in range(layer_count):
            conv = conv_op(num_channels_in if i == 0 else num_channels_out, num_channels_out, kernel_size=kernel_size, padding=padding)
            relu = nn.PReLU()
            layers.append(conv)
            layers.append(relu)

        self.vnet_layer = nn.Sequential(*layers)


    def forward(self, x):
        return self.vnet_layer(x)

class VNet(nn.Module):
    def __init__(self, config, tnet=None):

        super(VNet, self).__init__()


        self.config = config
        ndims = self.config.ndims
        num_input_channels = self.config.num_input_channels
        batch_norm = self.config.more_options.batch_norm
        stn_mode = self.config.more_options.STN

        if ndims == 2:
            ConvLayer = nn.Conv2d
            ConvTransposeLayer = nn.ConvTranspose2d
            self.affine_grid = stn2d.affine_grid
            self.grid_sample  = stn2d.grid_sample
        elif ndims == 3:
            ConvLayer = nn.Conv3d
            ConvTransposeLayer = nn.ConvTranspose3d
            self.affine_grid = stn3d.affine_grid
            self.grid_sample = stn3d.grid_sample
        else:
            raise "Unet only works in 2 or 3 dimensions"

        first_layer_channels = self.config.first_layer_channels
        layer1 = VNetLayer(num_input_channels,
                           first_layer_channels,
                           ndims=ndims,
                           layer_count=1,
                           border_mode=self.config.border_mode,
                           batch_norm=batch_norm)

        res1 = ConvLayer(in_channels=num_input_channels,
                                out_channels=first_layer_channels,
                                kernel_size=1)
        # Down layers.
        down_layers = []
        for i in range(1, self.config.steps + 1):

            downconv = ConvLayer(in_channels=first_layer_channels * 2**(i - 1),
                                 out_channels=first_layer_channels * 2**(i - 1),
                                 kernel_size=2,
                                 stride=2)


            lyr = VNetLayer(first_layer_channels * 2**(i - 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            layer_count= 2 if i == 1 else 3,
                            border_mode=self.config.border_mode,
                            batch_norm=batch_norm)

            res_matching = ConvLayer(in_channels=first_layer_channels * 2**(i - 1),
                             out_channels=first_layer_channels * 2**i,
                             kernel_size=1)


            down_layers.append((downconv, lyr, res_matching))


        if self.config.more_options.prior is not None:
            prior_channels = self.config.more_options.prior.shape[1]
        else:
            prior_channels = 0

        # Up layers
        up_layers = []
        for i in range(self.config.steps - 1, -1, -1):
            # Up-convolution
            upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1),
                                        out_channels=first_layer_channels * 2**i,
                                        kernel_size=2,
                                        stride=2)
            lyr = VNetLayer(first_layer_channels * 2**(i + 1) + prior_channels,
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            layer_count= i + 1 if i <= 1 else 3,
                            border_mode=self.config.border_mode,
                            batch_norm=batch_norm)

            up_layers.append((upconv, lyr))



        final_layer = ConvLayer(in_channels=first_layer_channels + prior_channels,
                                out_channels=self.config.num_classes,
                                kernel_size=1)

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.first_layer = layer1
        self.first_residual_matching = res1
        self.down = nn.Sequential(*chain(*down_layers))
        self.up = nn.Sequential(*chain(*up_layers))
        self.final_layer = final_layer


        if stn_mode == 3 or stn_mode == 4:

            first_layer_channels = 8 # misleading name. at the time of STN 1, it was a seperate network. so this determined the number of kernels in first layer.

            self.do_mid_st = self.stn_list

            if stn_mode == 3: # just angle estimation, no stn
                # self.fc_loc_laye1_size = first_layer_channels * 2 ** self.config.steps * 6 * 6 * 6
                self.fc_loc_laye1_size = first_layer_channels * 2 ** self.config.steps * 2 * 11 * 11

                # self.fc_loc_laye1_size = first_layer_channels * 2 ** self.config.steps * 9 * 9
                self.localization = lambda x: x[:, :first_layer_channels * 2 ** self.config.steps]
            else:
                assert False, 'invalid STN option'

            fc_layers = [
                nn.Linear(self.fc_loc_laye1_size, self.fc_loc_laye1_size // 16),
                nn.ReLU(True),
                nn.Linear(self.fc_loc_laye1_size // 16, self.fc_loc_laye1_size // 64),
                nn.ReLU(True),
                nn.Linear(self.fc_loc_laye1_size // 64, self.config.more_options.theta_param_count)
            ]
            self.fc_loc = nn.Sequential(*fc_layers)

            if ndims == 3 and self.config.more_options.prior is not None:
                self.prior = self.config.more_options.prior.double().repeat(self.config.batch_size, 1, 1, 1, 1).cuda()
                self.upsample_mode = 'trilinear'
                self.merge_prior = crop_and_merge

                _, _,D, H, W = self.prior.shape
                self.center = tuple((D//2, H//2, W//2))

            elif ndims == 2 and self.config.more_options.prior is not None:
                self.prior = self.config.more_options.prior.double().cuda()
                self.upsample_mode = 'bilinear'
                self.merge_prior = crop_and_merge

                _, _, H, W = self.prior.shape
                self.center = tuple((H//2, W//2))
            else:
                self.merge_prior = lambda x1,x2: x2

        else:
            self.do_mid_st = lambda x: torch.Tensor([0, -1, 0]).cuda()[None]


    # Spatial transformer network forward function for list
    @numpytorch
    def stn_list(self, x_list):

        # estimate parameters
        xs = self.localization(x_list[-1])
        xs = xs.view(-1, self.fc_loc_laye1_size)
        params = self.fc_loc(xs)

        return params

    @staticmethod
    def do_shift(x, coords):

        # scale
        theta = torch.eye(2,3)[None].repeat(x.shape[0], 1, 1).cuda().float()
        theta[:, 0, 0] = 0.145 / coords[:, 3]
        theta[:, 1, 1] = 0.145 / coords[:, 2]

        grid = F.affine_grid(theta, x.size()).cuda().double()
        x_shifted = F.grid_sample(x, grid, mode='bilinear')

        # shift
        theta = torch.eye(2,3)[None].repeat(x.shape[0], 1, 1).cuda().float()
        theta[:, 0, 2] =  -coords[:, 1]
        theta[:, 1, 2] =  -coords[:, 0]

        grid = F.affine_grid(theta, x_shifted.size()).cuda().double()
        x_shifted = F.grid_sample(x_shifted, grid, mode='bilinear')

        return x_shifted.float()

    @numpytorch
    def forward(self, input):

        # stn
        x = input

        # first layer
        x = self.first_layer(x) + self.first_residual_matching(x)
        down_outputs = [x]

        # down layers
        for (down_conv, unet_layer, res_matching) in self.down_layers:
            x = down_conv(x)
            x = unet_layer(x) + res_matching(x)
            down_outputs.append(x)

        # mid stn
        axis = self.do_mid_st(down_outputs)

        registered_atlas = stns.register_atlas(self.prior.double(), axis, input.shape)
        #### oriented_mask = patch_utils.get_patch(oriented_mask, (1, 3,) + (96, 96, 96), (0, 1,) + (135 // 2, 135 // 2, 135 // 2), mode='constant', copy=False).patch

        # oriented_mask_1 = UTNet.do_shift(self.prior[:, 0][:, None].repeat(input.shape[0], 1, 1, 1), axis[:, :4]).cuda()
        # oriented_mask_2 = UTNet.do_shift(self.prior[:, 0][:, None].repeat(input.shape[0], 1, 1, 1), axis[:, 4:]).cuda()
        # oriented_mask = torch.cat((oriented_mask_1, oriented_mask_2), dim=1).float()


        # up layers
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x_res = upconv_layer(x)
            x = crop_and_merge(down_output, x_res)
            x = crop_and_merge(F.upsample(registered_atlas, size=x.shape[2:], mode=self.upsample_mode), x) # mask
            x = unet_layer(x) + x_res

        x = crop_and_merge(F.upsample(registered_atlas, size=x.shape[2:], mode=self.upsample_mode), x) # mask
        x = self.final_layer(x)

        return x, axis

