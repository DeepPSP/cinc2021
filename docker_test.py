"""
"""
import traceback


if __name__ == "__main__":
    try:
        # in cfg.py
        from torch_ecg.torch_ecg.model_configs.ecg_crnn import ECG_CRNN_CONFIG
        from torch_ecg.torch_ecg.model_configs.cnn import (
            vgg_block_basic, vgg_block_mish, vgg_block_swish,
            vgg16, vgg16_leadwise,
            resnet_block_stanford, resnet_stanford,
            resnet_block_basic, resnet_bottle_neck,
            resnet, resnet_leadwise,
            multi_scopic_block,
            multi_scopic, multi_scopic_leadwise,
            dense_net_leadwise,
            xception_leadwise,
        )
        from torch_ecg.torch_ecg.model_configs.rnn import (
            lstm,
            attention,
            linear,
        )
        from torch_ecg.torch_ecg.model_configs.attn import (
            non_local,
            squeeze_excitation,
            global_context,
        )

        # in model.py
        from torch_ecg.torch_ecg.models.ecg_crnn import ECG_CRNN

        # in train.py
        from torch_ecg.torch_ecg.models._nets import BCEWithLogitsWithClassWeightLoss
        from torch_ecg.torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
        from torch_ecg.torch_ecg.utils.misc import (
            init_logger, get_date_str, dict_to_str, str2bool,
        )

        print("successfully import torch_ecg from the submodule!")
    except Exception as e:
        print("failed to import torch_ecg from the submodule!")
        traceback.print_exc()

    try:
        # in cfg.py
        from torch_ecg_bak.torch_ecg.model_configs.ecg_crnn import ECG_CRNN_CONFIG
        from torch_ecg_bak.torch_ecg.model_configs.cnn import (
            vgg_block_basic, vgg_block_mish, vgg_block_swish,
            vgg16, vgg16_leadwise,
            resnet_block_stanford, resnet_stanford,
            resnet_block_basic, resnet_bottle_neck,
            resnet, resnet_leadwise,
            multi_scopic_block,
            multi_scopic, multi_scopic_leadwise,
            dense_net_leadwise,
            xception_leadwise,
        )
        from torch_ecg_bak.torch_ecg.model_configs.rnn import (
            lstm,
            attention,
            linear,
        )
        from torch_ecg_bak.torch_ecg.model_configs.attn import (
            non_local,
            squeeze_excitation,
            global_context,
        )

        # in model.py
        from torch_ecg_bak.torch_ecg.models.ecg_crnn import ECG_CRNN

        # in train.py
        from torch_ecg_bak.torch_ecg.models._nets import BCEWithLogitsWithClassWeightLoss
        from torch_ecg_bak.torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
        from torch_ecg_bak.torch_ecg.utils.misc import (
            init_logger, get_date_str, dict_to_str, str2bool,
        )

        print("successfully import torch_ecg from the backup folder!")
    except Exception as e:
        print("failed to import torch_ecg from the backup folder!")
        traceback.print_exc()

    import torch
    cuda_is_available = " " if torch.cuda.is_available() else " not "
    print(f"torch version == {torch.__version__}")
    print(f"cuda is{cuda_is_available}available")
