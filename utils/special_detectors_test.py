"""
test of special detectors
"""
import os, json

from data_reader import CINC2021Reader
from cfg import BaseCfg
from .misc import list_sum
from .special_detectors import special_detectors


__all__ = ["test_sd",]


DR = CINC2021Reader(db_dir=BaseCfg.db_dir)


def test_sd(rec:str) -> dict:
    """
    """
    try:
        fs = DR.get_fs(rec)
        data = DR.load_data(rec)
        ann = DR.load_ann(rec)["diagnosis_scored"]["diagnosis_abbr"]
        cc = special_detectors(data, fs, rpeak_fn="xqrs")
        cc = [k.replace("is_", "") for k,v in cc.items() if v]
        ret_val = {"rec": rec, "label": ann, "pred": cc, "err": "False"}
    except:
        ret_val = {"rec": rec, "label": [], "pred": [], "err": "True"}
    return ret_val


if __name__ == "main":
    all_candidates = sorted(set(list_sum(
        DR.diagnoses_records_list[k] \
            for k in ["Brady", "STach", "SB", "LQRSV", "RAD", "LAD", "PR",]
    )))
    print(f"number of candidate records: {len(all_candidates)}")
    all_results = []
    # NOTE that multiprocessing is used inside the signal preprocessing function,
    # hence test_sd could not be parallel computed
    for idx, rec in enumerate(all_candidates):
        all_results.append(test_sd(rec))
        print(f"{idx+1}/{len(all_candidates)}", end="\r")

    save_fp = os.path.join(BaseCfg.log_dir, "special_detectors_test_results.json")
    with open(save_fp, "w") as f:
        json.dump(all_results, save_fp, ensure_ascii=False)
