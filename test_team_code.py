"""
"""

import os, sys
from copy import deepcopy
from typing import NoReturn

import numpy as np

from team_code import load_model, run_model
from helper_code import *
from cfg import TrainCfg, TrainCfg_ns


def test_load_model() -> NoReturn:
    """
    """
    for l in lead_sets:
        model = load_model(TrainCfg_ns.model_dir, l)
        print(f"n_leads = {len(l)}")
        print(model)
        print("#"*80 + "\n\n")


def test_run_model() -> NoReturn:
    """
    """
    base_dir = os.path.dirname(TrainCfg_ns.model_dir)
    for l in ["twelve", "six", "four", "three", "two",]:
        test_model(
            TrainCfg_ns.model_dir,
            os.path.join(base_dir, "test_data", f"{l}_leads"),
            os.path.join(TrainCfg_ns.log_dir, f"test_results_{l}_leads"),
        )


def test_model(model_directory, data_directory, output_directory):
    """
    the function from test_model.py
    """
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the outputs if it does not already exist.
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Identify the required lead sets.
    required_lead_sets = set()
    for i in range(num_recordings):
        header = load_header(header_files[i])
        leads = get_leads(header)
        sorted_leads = sort_leads(leads)
        required_lead_sets.add(sorted_leads)

    # Load models.
    leads_to_model = dict()
    print('Loading models...')
    for leads in required_lead_sets:
        model = load_model(model_directory, leads) ### Implement this function!
        leads_to_model[leads] = model

    # Run model for each recording.
    print('Running model...')

    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        leads = get_leads(header)
        sorted_leads = sort_leads(leads)

        # Apply model to recording.
        model = leads_to_model[sorted_leads]
        try:
            classes, labels, probabilities = run_model(model, header, recording) ### Implement this function!
        except Exception as e:
            print(e)
            print('... failed.')
            classes, labels, probabilities = list(), list(), list()

        # Save model outputs.
        recording_id = get_recording_id(header)
        head, tail = os.path.split(header_files[i])
        root, extension = os.path.splitext(tail)
        output_file = os.path.join(output_directory, root + '.csv')
        save_outputs(output_file, recording_id, classes, labels, probabilities)

    print('Done.')


if __name__ == "__main__":
    print("#"*80)
    print("test_load_model...")
    test_load_model()
    print("\n"*3)
    print("#"*80)
    print("test_run_model...")
    test_run_model()
