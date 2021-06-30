"""
test of special detectors
"""
import os, json
from typing import Sequence, Optional
from random import sample
import argparse

from data_reader import CINC2021Reader
from cfg import (
    BaseCfg, Standard12Leads,
    twelve_leads, six_leads, four_leads, three_leads, two_leads,
)
from .misc import list_sum
from .special_detectors import special_detectors


__all__ = ["test_sd",]


DR = CINC2021Reader(db_dir=BaseCfg.db_dir)


def test_sd(rec:str, leads:Sequence[str]=Standard12Leads, verbose:int=0, dr:Optional[type(CINC2021Reader)]=None) -> dict:
    """
    """
    try:
        ret_val = _test_sd(rec, leads, verbose, dr=dr)
    except:
        ret_val = {"rec": rec, "label": [], "pred": [], "err": "True"}
    return ret_val


def _test_sd(rec:str, leads:Sequence[str]=Standard12Leads, verbose:int=0, dr:Optional[type(CINC2021Reader)]=None) -> dict:
    """
    """
    _dr = dr or DR
    fs = _dr.get_fs(rec)
    data = _dr.load_data(rec, leads=list(leads))
    # ann = _dr.load_ann(rec)["diagnosis_scored"]["diagnosis_abbr"]
    ann = _dr.get_labels(rec, scored_only=True, fmt="a", normalize=True)
    cc = special_detectors(data, fs, rpeak_fn="xqrs", leads=list(leads), verbose=verbose)
    cc = [k.replace("is_", "") for k,v in cc.items() if v]
    ret_val = {"rec": rec, "label": ann, "pred": cc, "err": "False"}
    return ret_val


def get_parser() -> dict:
    """
    """
    description = "performing test using the special detectors"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--db-dir", type=str,
        help="directory of the CinC2021 database",
        dest="db_dir",
    )
    parser.add_argument(
        "-p", "--prop", type=float,
        help="proportion of the records used for testing, should be within ]0,1[",
        dest="proportion",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help=f"verbosity",
        dest="verbose",
    )

    args = vars(parser.parse_args())

    return args


if __name__ == "main":
    args = get_parser()
    size = args.get("proportion", 0.1)  # up to 1
    db_dir = args.get("db_dir", None)
    _dr = CINC2021Reader(db_dir) if db_dir else DR
    all_candidates = sorted(set(list_sum(
        sample(_dr.diagnoses_records_list[k], int(round(len(_dr.diagnoses_records_list[k])*size))) \
            for k in ["Brady", "STach", "SB", "LQRSV", "RAD", "LAD", "PR",]
    )))
    print(f"number of candidate records: {len(all_candidates)}")
    all_results = {
        "twelve_leads": [],
        "six_leads": [],
        "four_leads": [],
        "three_leads": [],
        "two_leads": [],
    }
    leads = {
        "twelve_leads": list(twelve_leads),
        "six_leads": list(six_leads),
        "four_leads": list(four_leads),
        "three_leads": list(three_leads),
        "two_leads": list(two_leads),
    }
    # NOTE that multiprocessing is used inside the signal preprocessing function,
    # hence test_sd could not be parallel computed
    for k, l in all_results.items():
        for idx, rec in enumerate(all_candidates):
            l.append(test_sd(rec, leads=leads[k], dr=_dr))
            print(f"{k} --- {idx+1}/{len(all_candidates)}", end="\r")
        print("\n")

    os.makedirs(BaseCfg.log_dir, exist_ok=True)
    save_fp = os.path.join(BaseCfg.log_dir, "special_detectors_test_results.json")
    with open(save_fp, "w") as f:
        json.dump(all_results, f, ensure_ascii=False)
