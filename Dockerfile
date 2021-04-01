FROM wenh06/cinc2021_docker:pytorch1.6.0-cuda10.1-cudnn7-devel

# NOTE: The GPU provided by the Challenge is  GPU Tesla T4 with nvidiaDriverVersion: 418.40.04
# by checking https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# and https://download.pytorch.org/whl/torch_stable.html
# one should use 1.6.0-cuda10.1


## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet


RUN python docker_test.py


RUN python test_model.py ./saved_models ./test_data/twelve_leads ./log/test_12leads
RUN python test_model.py ./saved_models ./test_data/six_leads ./log/test_6leads
RUN python test_model.py ./saved_models ./test_data/three_leads ./log/test_3leads
RUN python test_model.py ./saved_models ./test_data/two_leads ./log/test_2leads



# commands to run test with docker container:

# cd ~/Jupyter/temp/cinc2021_docker_test/data/
# cp E075* ../test_data
# cd ~/Jupyter/temp/cinc2021_docker_test/cinc2021/
# sudo docker build -t image .
# sudo docker run -it -v ~/Jupyter/temp/cinc2021_docker_test/model:/physionet/model -v ~/Jupyter/temp/cinc2021_docker_test/test_data:/physionet/test_data -v ~/Jupyter/temp/cinc2021_docker_test/test_outputs:/physionet/test_outputs -v ~/Jupyter/temp/cinc2021_docker_test/data:/physionet/training_data image bash

# python train_model.py training_data model
# python test_model.py model test_data test_outputs
