#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from train import train


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


    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    # TODO: add train(...)

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    # TODO: add train(...)

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    # TODO: add train(...)

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    # TODO: add train(...)

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
