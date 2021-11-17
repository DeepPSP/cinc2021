"""
(CRNN) models training

Training strategy:
------------------
1. the following pairs of classes will be treated the same:

2. the following classes will be determined by the special detectors:
    PR, LAD, RAD, LQRSV, Brady,
    (potentially) SB, STach

3. models will be trained for each tranche separatly:

4. one model will be trained using the whole dataset (consider excluding tranche C? good news is that tranche C mainly consists of "Brady" and "STach", which can be classified using the special detectors)
        
References: (mainly tips for faster and better training)
-----------
1. https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/
2. (optim) https://www.fast.ai/2018/07/02/adam-weight-decay/
3. (lr) https://spell.ml/blog/lr-schedulers-and-adaptive-optimizers-YHmwMhAAACYADm6F
4. more....
"""

import os, sys, time
from typing import Tuple, Dict, Any

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from torch.utils.data import DataLoader
from easydict import EasyDict as ED


from torch_ecg.torch_ecg.utils.trainer import BaseTrainer
from model import ECG_CRNN_CINC2021
from utils.scoring_metrics import evaluate_scores
from cfg import BaseCfg, TrainCfg, ModelCfg
from dataset import CINC2021

if BaseCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = torch.float64
else:
    _DTYPE = torch.float32


class CINC2021Trainer(BaseTrainer):
    """
    """
    __name__ = "CINC2021Trainer"

    def run_one_step(self, *data:Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        data: tuple of Tensors,
            the data to be processed for training one step (batch),
            should be of the following order:
            signals, labels, *extra_tensors

        Returns
        -------
        preds: Tensor,
            the predictions of the model for the given data
        labels: Tensor,
            the labels of the given data
        """
        signals, labels = data
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        preds = model(signals)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, data_loader:DataLoader) -> Dict[str, float]:
        """
        """
        model.eval()

        all_scalar_preds = []
        all_bin_preds = []
        all_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preds, bin_preds = self._model.inference(signals)
            all_scalar_preds.append(preds)
            all_bin_preds.append(bin_preds)
        
        all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
        all_bin_preds = np.concatenate(all_bin_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        classes = data_loader.dataset.all_classes

        if debug:
            msg = f"all_scalar_preds.shape = {all_scalar_preds.shape}, all_labels.shape = {all_labels.shape}"
            self.log_manager.log_message(msg)
            head_num = 5
            head_scalar_preds = all_scalar_preds[:head_num,...]
            head_bin_preds = all_bin_preds[:head_num,...]
            head_preds_classes = [np.array(classes)[np.where(row)] for row in head_bin_preds]
            head_labels = all_labels[:head_num,...]
            head_labels_classes = [np.array(classes)[np.where(row)] for row in head_labels]
            for n in range(head_num):
                msg = textwrap.dedent(f"""
                ----------------------------------------------
                scalar prediction:    {[round(n, 3) for n in head_scalar_preds[n].tolist()]}
                binary prediction:    {head_bin_preds[n].tolist()}
                labels:               {head_labels[n].astype(int).tolist()}
                predicted classes:    {head_preds_classes[n].tolist()}
                label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """)
                self.log_manager.log_message(msg)

        auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric = \
            evaluate_scores(
                classes=classes,
                truth=all_labels,
                scalar_pred=all_scalar_preds,
                binary_pred=all_bin_preds,
            )
        eval_res = dict(
            auroc=auroc,
            auprc=auprc,
            accuracy=accuracy,
            f_measure=f_measure,
            f_beta_measure=f_beta_measure,
            g_beta_measure=g_beta_measure,
            challenge_metric=challenge_metric,
        )

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, for CinC2021, it is 0,
        """
        return 0


def get_args(**kwargs:Any):
    """
    """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2021",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-l", "--leads",
        type=int, default=12,
        help="number of leads",
        dest="n_leads")
    parser.add_argument(
        "-t", "--tranches",
        type=str, default="",
        help="the tranches for training",
        dest="tranches_for_training")
    parser.add_argument(
        "-b", "--batch-size",
        type=int, default=128,
        help="the batch size for training",
        dest="batch_size")
    parser.add_argument(
        "-c", "--cnn-name",
        type=str, default="multi_scopic_leadwise",
        help="choice of cnn feature extractor",
        dest="cnn_name")
    parser.add_argument(
        "-r", "--rnn-name",
        type=str, default="none",
        help="choice of rnn structures",
        dest="rnn_name")
    parser.add_argument(
        "-a", "--attn-name",
        type=str, default="se",
        help="choice of attention structures",
        dest="attn_name")
    parser.add_argument(
        "--keep-checkpoint-max", type=int, default=20,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max")
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="train with more debugging information",
        dest="debug")
    
    args = vars(parser.parse_args())

    cfg.update(args)
    
    return ED(cfg)


if __name__ == "__main__":
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tranches = train_config.tranches_for_training
    if tranches:
        classes = train_config.tranche_classes[tranches]
    else:
        classes = train_config.classes

    if train_config.n_leads == 12:
        model_config = deepcopy(ModelCfg.twelve_leads)
    elif train_config.n_leads == 6:
        model_config = deepcopy(ModelCfg.six_leads)
    elif train_config.n_leads == 4:
        model_config = deepcopy(ModelCfg.four_leads)
    elif train_config.n_leads == 3:
        model_config = deepcopy(ModelCfg.three_leads)
    elif train_config.n_leads == 2:
        model_config = deepcopy(ModelCfg.two_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    ECG_CRNN_CINC2021.__DEBUG__ = False
    model = ECG_CRNN_CINC2021(
        classes=classes,
        n_leads=train_config.n_leads,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=device)

    trainer = CINC2021Trainer(
        model=model,
        dataset_cls=CINC2021,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )

    try:
        best_model_state_dict = trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
