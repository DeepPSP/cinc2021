"""
test of special detectors
"""
import os, json
from typing import Sequence
from random import sample

from data_reader import CINC2021Reader
from cfg import (
    BaseCfg, Standard12Leads,
    twelve_leads, six_leads, three_leads, two_leads,
)
from .misc import list_sum
from .special_detectors import special_detectors


__all__ = ["test_sd",]


DR = CINC2021Reader(db_dir=BaseCfg.db_dir)


def test_sd(rec:str, leads:Sequence[str]=Standard12Leads, verbose:int=0) -> dict:
    """
    """
    if verbose == 0:
        try:
            ret_val = _test_sd(rec, leads)
        except:
            ret_val = {"rec": rec, "label": [], "pred": [], "err": "True"}
    else:
        ret_val = _test_sd(rec, leads, verbose)
    return ret_val


def _test_sd(rec:str, leads:Sequence[str]=Standard12Leads, verbose:int=0) -> dict:
    """
    """
    fs = DR.get_fs(rec)
    data = DR.load_data(rec, leads=list(leads))
    # ann = DR.load_ann(rec)["diagnosis_scored"]["diagnosis_abbr"]
    ann = DR.get_labels(rec, scored_only=True, fmt="a", normalize=True)
    cc = special_detectors(data, fs, rpeak_fn="xqrs", leads=list(leads), verbose=verbose)
    cc = [k.replace("is_", "") for k,v in cc.items() if v]
    ret_val = {"rec": rec, "label": ann, "pred": cc, "err": "False"}
    return ret_val


if __name__ == "main":
    size = 0.1  # up to 1
    all_candidates = sorted(set(list_sum(
        sample(DR.diagnoses_records_list[k], int(round(len(DR.diagnoses_records_list[k])*size))) \
            for k in ["Brady", "STach", "SB", "LQRSV", "RAD", "LAD", "PR",]
    )))
    print(f"number of candidate records: {len(all_candidates)}")
    all_results = {
        "twelve_leads": [],
        "six_leads": [],
        "three_leads": [],
        "two_leads": [],
    }
    leads = {
        "twelve_leads": list(twelve_leads),
        "six_leads": list(six_leads),
        "three_leads": list(three_leads),
        "two_leads": list(two_leads),
    }
    # NOTE that multiprocessing is used inside the signal preprocessing function,
    # hence test_sd could not be parallel computed
    for k, l in all_results.items():
        for idx, rec in enumerate(all_candidates):
            l.append(test_sd(rec, leads=leads[k]))
            print(f"{k} --- {idx+1}/{len(all_candidates)}", end="\r")
        print("\n")

    os.makedirs(BaseCfg.log_dir, exist_ok=True)
    save_fp = os.path.join(BaseCfg.log_dir, "special_detectors_test_results.json")
    with open(save_fp, "w") as f:
        json.dump(all_results, f, ensure_ascii=False)
