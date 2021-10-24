# Warsaw University of Technology

from layers.eca_block import ECABasicBlock
from models.minkgl import MinkHead, MinkTrunk, MinkGL


from models.minkloc import MinkLoc
from third_party.minkloc3d.minkloc import MinkLoc3D
from misc.utils import ModelParams


def model_factory(model_params: ModelParams):
    in_channels = 1

    if model_params.model == 'MinkLoc':
        model = MinkLoc(in_channels=in_channels,  feature_size=model_params.feature_size,
                        output_dim=model_params.output_dim, planes=model_params.planes,
                        layers=model_params.layers, num_top_down=model_params.num_top_down,
                        conv0_kernel_size=model_params.conv0_kernel_size, block=model_params.block,
                        pooling_method=model_params.pooling)
    elif model_params.model == 'MinkLoc3D':
        model = MinkLoc3D()
    elif 'egonn' in model_params.model:
        model = create_egonn_model(model_params)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

    return model


def create_egonn_model(model_params: ModelParams):
    model_name = model_params.model

    global_normalize = False
    local_normalize = True

    if model_name == 'egonn':
        # THIS IS OUR BEST MODEL
        block = ECABasicBlock
        planes = [32, 64, 64, 128, 128, 128, 128]
        layers = [1, 1, 1, 1, 1, 1, 1]

        global_in_levels = [5, 6, 7]
        global_map_channels = 128
        global_descriptor_size = 256

        local_in_levels = [3, 4]
        local_map_channels = 64
        local_descriptor_size = 128

    elif model_name == 'egonn_global':
        # THIS IS OUR BEST MODEL
        block = ECABasicBlock
        planes = [32, 64, 64, 128, 128, 128, 128]
        layers = [1, 1, 1, 1, 1, 1, 1]

        global_in_levels = [5, 6, 7]
        global_map_channels = 128
        global_descriptor_size = 256

        local_in_levels = []
        local_map_channels = None
        local_descriptor_size = None

    elif model_name == 'egonn_nofpn1':
        # Same as egonn but without FPN
        block = ECABasicBlock
        planes = [32, 64, 64, 128, 128]
        layers = [1, 1, 1, 1, 1]

        global_in_levels = [5]
        global_map_channels = 128
        global_descriptor_size = 256

        local_in_levels = [3]
        local_map_channels = 64
        local_descriptor_size = 128

    elif model_name == 'egonn_nofpn2':
        # Same as egonn but without FPN (second variant)
        block = ECABasicBlock
        planes = [32, 64, 64, 128, 128, 128, 128]
        layers = [1, 1, 1, 1, 1, 1, 1]

        global_in_levels = [7]
        global_map_channels = 128
        global_descriptor_size = 256

        local_in_levels = [4]
        local_map_channels = 64
        local_descriptor_size = 128

    else:
        raise NotImplementedError(f'Unknown model: {model_name}')

    # Planes list number of channels for level 1 and above
    global_in_channels = [planes[i-1] for i in global_in_levels]
    head_global = MinkHead(global_in_levels, global_in_channels, global_map_channels)

    if len(local_in_levels) > 0:
        local_in_channels = [planes[i-1] for i in local_in_levels]
        head_local = MinkHead(local_in_levels, local_in_channels, local_map_channels)
    else:
        head_local = None

    min_out_level = len(planes)
    if len(global_in_levels) > 0:
        min_out_level = min(min_out_level, min(global_in_levels))
    if len(local_in_levels) > 0:
        min_out_level = min(min_out_level, min(local_in_levels))

    trunk = MinkTrunk(in_channels=1, planes=planes, layers=layers, conv0_kernel_size=5, block=block,
                      min_out_level=min_out_level)

    net = MinkGL(trunk, local_head=head_local, local_descriptor_size=local_descriptor_size,
                 local_normalize=local_normalize, global_head=head_global,
                 global_descriptor_size=global_descriptor_size, global_pool_method='GeM',
                 global_normalize=global_normalize, quantizer=model_params.quantizer)

    return net