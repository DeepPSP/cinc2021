#!/bin/sh
black . --extend-exclude .ipynb -v --exclude "/(build|dist|torch\_ecg|torch\_ecg\_bak|official\_baseline\_classifier|official\_scoring\_metric|helper\_code\.py|run\_model\.py|train\_model\.py|evaluate\_model\.py)/"
flake8 . --count --ignore="E501 W503 E203 F841 E402 E731" --show-source --statistics --exclude=./.*,build,dist,official*,torch_ecg,torch_ecg_bak,pantompkins.py,helper_code.py,run_model.py,train_model.py,evaluate_model.py,*.ipynb
