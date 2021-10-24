# Warsaw University of Technology

import torch
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from models.minkfpn import MinkFPN
import layers.pooling as pooling
from layers.senet_block import SEBasicBlock, SEBottleneck
from layers.eca_block import ECABasicBlock


class MinkLoc(torch.nn.Module):
    def __init__(self, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size,
                 block='BasicBlock', pooling_method='GeM'):
        # block: Type of the network building block: BasicBlock or SEBasicBlock
        # add_linear_layers: Add linear layers at the end
        # dropout_p: dropout probability (None = no dropout)

        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size  # Size of local features produced by local feature extraction block
        self.output_dim = output_dim  # Dimensionality of the global descriptor produced by pooling layer
        self.block = block

        if block == 'BasicBlock':
            block_module = BasicBlock
        elif block == 'Bottleneck':
            block_module = Bottleneck
        elif block == 'SEBasicBlock':
            block_module = SEBasicBlock
        elif block == 'ECABasicBlock':
            block_module = ECABasicBlock
        else:
            raise NotImplementedError('Unsupported network block: {}'.format(block))

        self.pooling_method = pooling_method
        self.backbone = MinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                conv0_kernel_size=conv0_kernel_size, block=block_module, layers=layers, planes=planes)
        self.pooling = pooling.PoolingWrapper(pool_method=pooling_method, in_dim=self.feature_size,
                                              output_dim=output_dim)
        self.pooled_feature_size = self.pooling.output_dim  # Number of channels returned by pooling layer

    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        x = self.backbone(x)

        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)

        x = self.pooling(x)
        if x.dim() == 3 and x.shape[2] == 1:
            # Reshape (batch_size,
            x = x.flatten(1)

        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.pooled_feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.pooled_feature_size)
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor
        return {'global': x}

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print('Backbone parameters: {}'.format(n_params))
        print('Backbone building block: {}'.format(self.block))
        print('Pooling method: {}'.format(self.pooling_method))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print('Pooling parameters: {}'.format(n_params))
        print('# channels from feature extraction block: {}'.format(self.feature_size))
        print('# channels from pooling block: {}'.format(self.pooled_feature_size))
        print('# output channels : {}'.format(self.output_dim))
