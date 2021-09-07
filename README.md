# [PhysioNet/CinC Challenge 2021](https://physionetchallenges.github.io/2021/)
Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021


## Conference Website and Conference Programme
[Website](http://www.cinc2021.org/), [Programme](https://www.cinc.org/2021/Program/accepted/PreliminaryProgram.html), [Poster](/images/CinC2021_poster.pdf)


## Data Preparation
One can download training data from [GCP](https://console.cloud.google.com/storage/browser/physionetchallenge2021-public-datasets),
and use `python prepare_dataset -i {data_directory} -v` to prepare the data for training


## Deep Models
Deep learning models are constructed using [torch_ecg](https://github.com/DeepPSP/torch_ecg), which has already been added as a submodule.


## [Images](/images/)

- 2 typical training processes

![2 typical training processes](/images/train.svg)

- "Confusion Matrix" of a typical model on the train-validation set

![cm_bin](/images/confusion-matrix-multi-scopic-ncr.svg)

The "Confusion Matrix" is quoted since it is not really a confusion matrix (the classification is multi-label classification). Its computation can be found [here](https://github.com/DeepPSP/cinc2021/blob/master/gather_results.py#L122). The diagonal are "true positives", the off-diagonal are "false positives". The "false negatives" are not reflected on this figure. For more matrix plot, ref. [Images](/images/)


## Digest of Top Models
to be updated after the conference


## References:
TO add....


## Misc
[Link](https://github.com/DeepPSP/cinc2020) to the unsuccessful attemps for CinC2020 of the previous year.
