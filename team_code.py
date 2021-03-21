#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from copy import deepcopy

import torch

from train import train
from cfg import TrainCfg, ModelCfg, SpecialDetectorCfg
from utils.special_detectors import special_detectors
from utils.utils_nn import extend_predictions
from utils.misc import get_date_str, dict_to_str, init_logger
from model import ECG_CRNN_CINC2021


twelve_lead_model_filename = '12_lead_model.pth'
six_lead_model_filename = '6_lead_model.pth'
three_lead_model_filename = '3_lead_model.pth'
two_lead_model_filename = '2_lead_model.pth'

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)


    # configs and logger
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = data_directory
    train_config.model_dir = model_directory

    tranches = train_config.tranches_for_training
    if tranches:
        train_classes = train_config.tranche_classes[tranches]
    else:
        train_classes = train_config.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = init_logger(log_dir=train_config.log_dir, verbose=2)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f"Using device {device}")
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f"with configuration\n{dict_to_str(train_config)}")


    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    train_config.leads = twelve_leads
    train_config.n_leads = len(train_config.leads)
    # filename = os.path.join(model_directory, twelve_lead_model_filename)
    train_config.final_model_name = twelve_lead_model_filename
    model_config = deepcopy(ModelCfg.twelve_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    model = ECG_CRNN_CINC2021(
        classes=train_classes,
        n_leads=train_config.n_leads,
        # input_len=config.input_len,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    model.__DEBUG__ = False

    train(
        model=model,
        config=train_config,
        device=device,
        logger=logger,
        debug=train_config.debug,
    )


    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    train_config.leads = six_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = six_lead_model_filename
    model_config = deepcopy(ModelCfg.six_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    model = ECG_CRNN_CINC2021(
        classes=train_classes,
        n_leads=train_config.n_leads,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    model.__DEBUG__ = False

    train(
        model=model,
        config=train_config,
        device=device,
        logger=logger,
        debug=train_config.debug,
    )

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    train_config.leads = three_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = three_lead_model_filename
    model_config = deepcopy(ModelCfg.three_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    model = ECG_CRNN_CINC2021(
        classes=train_classes,
        n_leads=train_config.n_leads,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    model.__DEBUG__ = False

    train(
        model=model,
        config=train_config,
        device=device,
        logger=logger,
        debug=train_config.debug,
    )

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    train_config.leads = two_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = twelve_lead_model_filename
    model_config = deepcopy(ModelCfg.two_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    model = ECG_CRNN_CINC2021(
        classes=train_classes,
        n_leads=train_config.n_leads,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    model.__DEBUG__ = False

    train(
        model=model,
        config=train_config,
        device=device,
        logger=logger,
        debug=train_config.debug,
    )

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    raise NotImplementedError

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    raise NotImplementedError

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    raise NotImplementedError

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    raise NotImplementedError

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    raise NotImplementedError

# Generic function for loading a model.
def load_model(filename):
    raise NotImplementedError

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    raise NotImplementedError
    # return classes, labels, probabilities




################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
# def get_features(header, recording, leads):
#     # Extract age.
#     age = get_age(header)
#     if age is None:
#         age = float('nan')

#     # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
#     sex = get_sex(header)
#     if sex in ('Female', 'female', 'F', 'f'):
#         sex = 0
#     elif sex in ('Male', 'male', 'M', 'm'):
#         sex = 1
#     else:
#         sex = float('nan')

#     # Reorder/reselect leads in recordings.
#     available_leads = get_leads(header)
#     indices = list()
#     for lead in leads:
#         i = available_leads.index(lead)
#         indices.append(i)
#     recording = recording[indices, :]

#     # Pre-process recordings.
#     adc_gains = get_adcgains(header, leads)
#     baselines = get_baselines(header, leads)
#     num_leads = len(leads)
#     for i in range(num_leads):
#         recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

#     # Compute the root mean square of each ECG lead signal.
#     rms = np.zeros(num_leads, dtype=np.float32)
#     for i in range(num_leads):
#         x = recording[i, :]
#         rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

#     return age, sex, rms
