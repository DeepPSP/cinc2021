# Images to illustrate data distributions, challenge results, etc.

Many of the images are created using data from [Results](/results/).

code are executed in the root directory of the challenge repository, say in `/home/wenh06/Jupyter/workspace/cinc2021/`

the [scored_classes_distribution.png](/images/scored_classes_distribution.png) is generated using the following code

```python
import matplotlib.pyplot as plt
from utils.scoring_aux_data import dx_mapping_scored

dx = dx_mapping_scored.sort_values("Total")
fig, ax = plt.subplots(figsize=(12,12))
ax.barh(dx.Abbreviation.values, dx.Total.values)
for i, v in enumerate(dx.Total.values):
    ax.text(v + 3, i-0.25, str(v))
ax.set_xlabel("# records")
ax.set_title("Scored Classes Distribution")
fig.savefig("./images/scored_classes_distribution.png", transparent=True, bbox_inches="tight")
```

the [train.pdf](/images/train.pdf) and [train.svg](/images/train.svg) are generated via
```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

from utils.misc import read_log_txt, read_event_scalars
from torch_ecg.benchmarks.database_reader.database_reader.utils.utils_signal import MovingAverage

default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

ml_12 = read_event_scalars("./results/20210714-12leads/events.out.tfevents.1626180427.Precision-7920-TowerOPT_ECG_CRNN_CINC2021_multi_scopic_leadwise_adamw_amsgrad_LR_0.001_BS_128_tranche_all")
ml_12_ncr = read_event_scalars("./results/20210827-12leads/events.out.tfevents.1630045554.Precision-7920-TowerOPT_ECG_CRNN_CINC2021_multi_scopic_leadwise_adamw_amsgrad_LR_0.001_BS_64_tranche_all")

fig, ax = plt.subplots(figsize=(16,12))
ax.plot(ml_12["train/loss"].step, MovingAverage(ml_12["train/loss"].value)._ema(), label="train-loss", color=default_color_cycle[0])
ax.plot(ml_12_ncr["train/loss"].step, MovingAverage(ml_12_ncr["train/loss"].value)._ema(), label="train-loss-ncr", color=default_color_cycle[1])
ax.text(ml_12["train/loss"].step.values[-1]-2000, ml_12["train/loss"].value.values[-1]-0.09, "train-loss", c=default_color_cycle[0], fontsize=16)
ax.text(ml_12_ncr["train/loss"].step.values[-1]-3500, ml_12_ncr["train/loss"].value.values[-1]-0.025, "train-loss-ncr", c=default_color_cycle[1], fontsize=16)
ax.set_ylim(0.2,1.2)
ax.set_xlabel("Steps (n.u.)",fontsize=22)
ax.set_ylabel("Loss (n.u.)",fontsize=22)
ax2 = ax.twinx()
ax2.plot(ml_12["train/challenge_metric"].step, ml_12["train/challenge_metric"].value, marker="x", markersize=8, label="train-cm", linestyle="solid", color=default_color_cycle[2])
ax2.plot(ml_12_ncr["train/challenge_metric"].step, ml_12_ncr["train/challenge_metric"].value, marker="x", markersize=8, label="train-cm-ncr", linestyle="dashed", color=default_color_cycle[3])
ax2.plot(ml_12["test/challenge_metric"].step, ml_12["test/challenge_metric"].value, marker="x", markersize=8, label="val-cm", linestyle="dotted", color=default_color_cycle[4])
ax2.plot(ml_12_ncr["test/challenge_metric"].step, ml_12_ncr["test/challenge_metric"].value, marker="x", markersize=8, label="val-cm-ncr", linestyle="dashdot", color=default_color_cycle[5])
ax2.set_ylim(0.35,0.9)
ax2.set_ylabel("Challenge Metric (n.u.)",fontsize=22)
ax2.legend(loc="best", fontsize=20)
rect_x = ml_12["test/challenge_metric"].step.values[-10]-200
rect_y = ml_12["test/challenge_metric"].value.values[-10:].min()-0.012
rect_width = ml_12["test/challenge_metric"].step.values[-1] - rect_x + 500
rect_height = ml_12["test/challenge_metric"].value.values[-10:].max() - rect_y + 0.016
rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, facecolor="r",alpha=0.3)
ax2.add_patch(rect)
ax2.text(rect_x+rect_width-1000, rect_y+rect_height-0.014, "early stopping", fontsize=16, c="r")
rect_x = ml_12_ncr["test/challenge_metric"].step.values[-10]-200
rect_y = ml_12_ncr["test/challenge_metric"].value.values[-10:].min()-0.012
rect_width = ml_12_ncr["test/challenge_metric"].step.values[-1] - rect_x + 500
rect_height = ml_12_ncr["test/challenge_metric"].value.values[-10:].max() - rect_y + 0.016
rect_ncr = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, facecolor="r",alpha=0.3)
ax2.add_patch(rect_ncr)
ax2.text(rect_x+rect_width-5000, rect_y+rect_height-0.014, "early stopping", fontsize=16, c="r")

fig.savefig("./images/train.svg", format="svg", dpi=1200, bbox_inches="tight")
fig.savefig("./images/train.pdf", format="pdf", dpi=1200, bbox_inches="tight")
```

Matrix plot of the mean value of scalar predictions of a model on the train-validation set
![matrix_mean_scalar](/images/mean-scalar-prediction-matrix-multi-scopic-ncr.svg)
Each column is the mean value of scalar (probability) predictions of the label to corresponding classes (on the left vertical axis).

Matrix plot of the standard deviation of scalar predictions of a model on the train-validation set
![matrix_std_scalar](/images/std-scalar-prediction-matrix-multi-scopic-ncr.svg)
