#!/bin/sh
black . --extend-exclude .ipynb -v --exclude "/(build|dist|torch_ecg|torch_ecg_bak|official_baseline_classifier|official_scoring_metric/|helper_code\.py|run_model\.py|train_model\.py|evaluate_model\.py|pantompkins\.py)"
flake8 . --count --ignore="E501 W503 E203 F841 E402 E731" --show-source --statistics --exclude=./.*,build,dist,official*,torch_ecg,torch_ecg_bak,pantompkins.py,helper_code.py,run_model.py,train_model.py,evaluate_model.py,*.ipynb
