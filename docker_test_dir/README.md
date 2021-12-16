# training data for docker image building

a tiny training set is generated via random sampling from the whole database of proportion 0.001


# test data for docker image building

test data generated using `python extract_leads_wfdb.py` from 100 samples in `test_data/twelve_leads/`.

docker image building shall fail if any of the following command fails
```
RUN python test_model.py ./saved_models ./docker_test_dir/test_data/twelve_leads ./log/test_12leads
RUN python test_model.py ./saved_models ./docker_test_dir/test_data/six_leads ./log/test_6leads
RUN python test_model.py ./saved_models ./docker_test_dir/test_data/four_leads ./log/test_4leads
RUN python test_model.py ./saved_models ./docker_test_dir/test_data/three_leads ./log/test_3leads
RUN python test_model.py ./saved_models ./docker_test_dir/test_data/two_leads ./log/test_2leads
```
