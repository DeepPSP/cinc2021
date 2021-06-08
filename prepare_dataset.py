"""
"""
import os, subprocess, shutil, argparse, tarfile, time
from glob import glob
from copy import deepcopy
from typing import Optional, Sequence, NoReturn

from cfg import BaseCfg


data_files = [
    "WFDB_CPSC2018.tar.gz",
    "WFDB_CPSC2018_2.tar.gz",
    "WFDB_StPetersburg.tar.gz",
    "WFDB_PTB.tar.gz",
    "WFDB_PTBXL.tar.gz",
    "WFDB_Ga.tar.gz",
    "WFDB_ShaoxingUniv.tar.gz",
    "WFDB_ChapmanShaoxing.tar.gz",
    "WFDB_Ningbo.tar.gz",
]

header_files = [
    "CPSC2018-Headers.tar.gz",
    "CPSC2018-2-Headers.tar.gz",
    "StPetersburg-Headers.tar.gz",
    "PTB-Headers.tar.gz",
    "PTB-XL-Headers.tar.gz",
    "Ga-Headers.tar.gz",
    "ShaoxingUniv_Headers.tar.gz",
    "ChapmanShaoxing-Headers.tar.gz",
    "Ningbo-Headers.tar.gz",
]

_tranches = "CPSC,CPSC_Extra,StPetersburg,PTB,PTB_XL,Georgia,CUSPHNFH".split(",")


def get_parser() -> dict:
    """
    """
    description = "Prepare the dataset, uncompressing the .tar.gz files, and replacing the header files."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input_directory", type=str,
        help="input directory, containing .tar.gz files of records and headers",
        dest="input_directory",
    )
    parser.add_argument(
        "-o", "--output_directory", type=str,
        help="output directory",
        dest="output_directory",
    )
    parser.add_argument(
        "-t", "--tranches", type=str,
        help=f"""list of tranches, a subset of {",".join(_tranches)}, separated by comma""",
        dest="tranches",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help=f"verbosity",
        dest="verbose",
    )

    args = vars(parser.parse_args())

    return args


def run(input_directory:str,
        output_directory:Optional[str]=None,
        tranches:Optional[Sequence[str]]=None,
        verbose:bool=False) -> NoReturn:
    """ finished, checked,

    Parameters
    ----------
    input_directory: str,
        directory containing the .tar.gz files of the records and headers
    output_directory: str, optional,
        directory to store the extracted records and headers, under specific organization,
        if not specified, defaults to `input_directory`
    tranches: sequence of str, optional,
        the tranches to extract
    verbose: bool, default False,
        printint verbosity

    NOTE: currently, for updating headers only, corresponding .tar.gz file of records should be presented
    """
    _dir = os.path.abspath(input_directory)
    # ShaoxingUniv (CUSPHNFH) is the union of ChapmanShaoxing and Ningbo
    if data_files[-3] in os.listdir(input_directory):
        flag_CUSPHNFH = False
        _data_files =  data_files[:-2]
        _header_files = header_files[:-2]
    else:
        flag_CUSPHNFH = True
        _data_files = deepcopy(data_files)
        _header_files = deepcopy(header_files)
    _data_files = \
        [os.path.basename(item) for item in glob(os.path.join(_dir, "WFDB_*.tar.gz")) if os.path.basename(item) in _data_files]
    _header_files = \
        [os.path.basename(item) for item in glob(os.path.join(_dir, "*Headers.tar.gz")) if os.path.basename(item) in _header_files]
    _output_directory = output_directory or input_directory
    assert all([header_files[data_files.index(item)] in _header_files for item in _data_files]), \
        "header files corresponding to some data files not found"

    if flag_CUSPHNFH:
        os.makedirs(os.path.join(_output_directory, "WFDB_CUSPHNFH"), exist_ok=True)

    acc = 0
    for i, df in enumerate(_data_files):
        if tranches and _tranches[data_files.index(df)] not in tranches:
            continue
        acc += 1
        if df in ["WFDB_ChapmanShaoxing.tar.gz", "WFDB_Ningbo.tar.gz",]:
            df_name = "WFDB_CUSPHNFH"
        else:
            df_name = df.replace(".tar.gz", "")
        if len(glob(os.path.join(_output_directory, df_name, "*.mat"))) > 0:
            pass
        else:
            with tarfile.open(os.path.join(_dir, df), "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        member.name = os.path.basename(member.name)
                        # header files will not be extracted,
                        # instead, they will be extracted from corresponding headers-only .tar.gz file
                        if os.path.splitext(member.name)[1] == ".hea":
                            continue
                        tar.extract(member, os.path.join(_output_directory, df_name))
                        if verbose:
                            print(f"extracted '{os.path.join(_output_directory, df_name, member.name)}'")
        print(f"finish extracting {df}")
        time.sleep(3)
        # corresponding header files
        hf = header_files[data_files.index(df)]
        with tarfile.open(os.path.join(_dir, hf), "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, os.path.join(_output_directory, df_name))
                    if verbose:
                        print(f"extracted '{os.path.join(_output_directory, df_name, member.name)}'")
        print(f"finish extracting {hf}")
        print(f"{df_name} done! --- {acc}/{len(tranches) if tranches else len(_data_files)}")
        if i < len(_data_files) - 1:
            time.sleep(3)



if __name__ == "__main__":
    # usage example:
    # python prepare_dataset.py --help (check for details of arguments)
    # python prepare_dataset.py -i "E:/Data/CinC2021/" -t "PTB,Georgia" -v
    args = get_parser()
    input_directory = args.get("input_directory", BaseCfg.db_dir)
    output_directory = args.get("output_directory", None)
    tranches = args.get("tranches", None)
    verbose = args.get("verbose", False)
    if tranches:
        tranches = tranches.split(",")
    run(input_directory, output_directory, tranches, verbose)
