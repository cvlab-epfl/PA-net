import numpy as np
class NetworkConfig(object):

    def __init__(self, steps, first_layer_channels, num_classes, num_input_channels, two_sublayers, ndims, border_mode,  seed, batch_size, config):

        if border_mode not in ['valid', 'same']:
            raise ValueError("`border_mode` not in ['valid', 'same']")

        self.steps = steps
        self.first_layer_channels = first_layer_channels
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.two_sublayers = two_sublayers
        self.ndims = ndims
        self.border_mode = border_mode
        self.seed = seed
        self.batch_size = batch_size

        border = 4 if self.two_sublayers else 2
        if self.border_mode == 'same':
            border = 0
        else:
            raise Exception('not supported anymore')
        self.first_step = lambda x: x - border
        self.rev_first_step = lambda x: x + border
        self.down_step = lambda x: (x - 1) // 2 + 1 - border
        self.rev_down_step = lambda x: (x + border) * 2
        self.up_step = lambda x: (x * 2) - border
        self.rev_up_step = lambda x: (x + border - 1) // 2 + 1
        self.config = config

    def __getstate__(self):
        return [self.steps, self.first_layer_channels, self.num_classes, self.two_sublayers, self.ndims, self.border_mode]

    def __setstate__(self, state):
        return self.__init__(*state)

    def __repr__(self):
        return "{0.__class__.__name__!s}(steps={0.steps!r}, first_layer_channels={0.first_layer_channels!r}, " \
                "num_classes={0.num_classes!r}, num_input_channels={0.num_input_channels!r}, "\
                "two_sublayers={0.two_sublayers!r}, ndims={0.ndims!r}, "\
                "border_mode={0.border_mode!r})".format(self)

    def in_out_shape(self, out_shape_lower_bound):
        """
        Compute the best combination of input/output shapes given the
        desired lower bound for the shape of the output
        """
        shape = np.asarray(out_shape_lower_bound)

        for i in range(self.steps):
            shape = self.rev_up_step(shape)

        # Best input shape
        for i in range(self.steps):
            shape = self.rev_down_step(shape)

        shape = self.rev_first_step(shape)

        return tuple(shape)

