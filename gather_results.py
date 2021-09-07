"""
"""

import os, re, time
from typing import Sequence, NoReturn, Optional, Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from torch_ecg.torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from model import ECG_CRNN_CINC2021
from dataset import CINC2021
from trainer import evaluate
from cfg import BaseCfg


__all__ = ["gather_from_checkpoint", "plot_confusion_matrix"]


def plot_confusion_matrix(cm:np.ndarray, classes:Sequence[str],
                          normalize:bool=False,
                          title:Optional[str]=None,
                          cmap:mpl.colors.Colormap=plt.cm.Blues,
                          fmt:str="svg") -> Any:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized Confusion Matrix"
            save_name = f"normalized_cm_{int(time.time())}.{fmt}"
        else:
            title = "Confusion Matrix"
            save_name = f"not_normalized_cm_{int(time.time())}.{fmt}"
    else:
        save_name = re.sub("[\s_-]+", "-", title.lower().replace(" ", "-")) + f".{fmt}"

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
        )
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("Label",fontsize=18)
    ax.set_ylabel("Predicted",fontsize=18)
    ax.tick_params(axis = "both", which = "major", labelsize = 13)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    text_fmt = ".2f" if (normalize or cm.dtype=="float") else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], text_fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    fig.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(BaseCfg.log_dir, save_name), format=fmt, dpi=1200, bbox_inches="tight")
    
    return ax


def gather_from_checkpoint(path:str, fmt:str="svg") -> NoReturn:
    """
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model = ECG_CRNN_CINC2021.from_checkpoint(path, device=device)
    print(f"model loaded from {path}")
    ds = CINC2021(ckpt["train_config"], training=False)
    dl =  DataLoader(
        dataset=ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    all_scalar_preds = []
    all_bin_preds = []
    all_labels = []

    start = time.time()
    print(f"start evaluating the model on the train-validation set...")
    for signals, labels in dl:
        signals = signals.to(device=device)
        labels = labels.numpy()
        all_labels.append(labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preds, bin_preds = model.inference(signals)
        all_scalar_preds.append(preds)
        all_bin_preds.append(bin_preds)

    all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
    all_bin_preds = np.concatenate(all_bin_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    classes = dl.dataset.all_classes
    
    print(f"evaluation used {time.time()-start:.2f} seconds")
    print("start computing the confusion matrix from the binary predictions")
    
    cm_bin = np.zeros((len(classes),len(classes)), dtype="int")
    for idx in range(all_labels.shape[0]):
        lb = set(all_labels[idx].nonzero()[0].tolist())
        bp = set(all_bin_preds[idx].nonzero()[0].tolist())
        for i in lb.intersection(bp):
            cm_bin[i,i] += 1
        for i in bp - lb:
            for j in lb:
                cm_bin[j,i] += 1
        print(f"{idx+1} / {len(all_labels)}", end="\r")
    title = f"""Confusion Matrix - {ckpt["train_config"]["cnn_name"].replace("_", "-")}"""
    if len(ckpt["train_config"]["special_classes"]) == 0:
        title += " - NCR"
    plot_confusion_matrix(
        cm=cm_bin,
        classes=classes,
        title=title,
        fmt=fmt,
    )

    print("start computing the ``confusion`` matrix from the scalar predictions")
    cm_scalar = {idx: [] for idx in range(len(classes))}

    for idx in range(all_labels.shape[0]):
        for i in all_labels[idx].nonzero()[0]:
            cm_scalar[int(i)].append(all_scalar_preds[idx])
        print(f"{idx+1} / {len(all_labels)}", end="\r")
    scalar_mean = {idx: np.mean(np.column_stack(v), axis=1) for idx,v in cm_scalar.items()}
    scalar_std = {idx: np.std(np.column_stack(v), axis=1) for idx,v in cm_scalar.items()}
    cm_scalar_mean = np.zeros((len(classes),len(classes)))
    cm_scalar_std = np.zeros((len(classes),len(classes)))
    for idx, v in scalar_mean.items():
        cm_scalar_mean[idx,...] = v
    for idx, v in scalar_std.items():
        cm_scalar_std[idx,...] = v
    
    title = f"""Mean Scalar Prediction Matrix - {ckpt["train_config"]["cnn_name"].replace("_", "-")}"""
    if len(ckpt["train_config"]["special_classes"]) == 0:
        title += " - NCR"
    plot_confusion_matrix(
        cm=cm_scalar_mean,
        classes=classes,
        title=title,
        fmt=fmt,
    )

    title = f"""STD Scalar Prediction Matrix - {ckpt["train_config"]["cnn_name"].replace("_", "-")}"""
    if len(ckpt["train_config"]["special_classes"]) == 0:
        title += " - NCR"
    plot_confusion_matrix(
        cm=cm_scalar_mean,
        classes=classes,
        title=title,
        fmt=fmt,
    )        
