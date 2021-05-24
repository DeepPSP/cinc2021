"""
"""
import os, subprocess, shutil, argparse, tarfile, time
from glob import glob
from copy import deepcopy
from typing import Optional, NoReturn

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
    "Ga-Headers.tar.gz",
    "PTB-Headers.tar.gz",
    "PTB-XL-Headers.tar.gz",
    "ShaoxingUniv_Headers.tar.gz",
    "ChapmanShaoxing-Headers.tar.gz",
    "Ningbo-Headers.tar.gz",
]


def get_parser() -> dict:
    description = "Prepare the dataset, uncompressing the .tar.gz files, and replacing the header files."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_directory", type=str)
    parser.add_argument("-o", "--output_directory", type=str)

    args = vars(parser.parse_args())

    return args


def run(input_directory:str, output_directory:Optional[str]=None) -> NoReturn:
    """
    """
    _dir = os.path.abspath(input_directory)
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
    print(_data_files)
    print(_header_files)
    assert all([header_files[data_files.index(item)] in _header_files for item in _data_files]), \
        "header files corresponding to some data files not found"

    if flag_CUSPHNFH:
        os.makedirs(os.path.join(_output_directory, "WFDB_CUSPHNFH"), exist_ok=True)

    headers_tmp = os.path.join(input_directory, "headers_tmp")
    if os.path.exists(headers_tmp):
        shutil.rmtree(headers_tmp)
    os.makedirs(headers_tmp, exist_ok=True)

    for i, df in enumerate(_data_files):
        if df in ["WFDB_ChapmanShaoxing.tar.gz", "WFDB_Ningbo.tar.gz",]:
            with tarfile.open(os.path.join(_dir, df), "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        member.name = os.path.basename(member.name)
                        tar.extract(member, os.path.join(_output_directory, "WFDB_CUSPHNFH"))
            df_name = "WFDB_CUSPHNFH"
        else:
            with tarfile.open(os.path.join(_dir, df), "r:gz") as tar:
                tar.extractall(_output_directory)
            df_name = df.replace(".tar.gz", "")
        print(f"finish extracting {df}")
        # corresponding header files
        hf = header_files[data_files.index(df)]
        with tarfile.open(os.path.join(_dir, hf), "r:gz") as tar:
            tar.extractall(headers_tmp)
        print(f"finish extracting {hf}")
        # remove old headers
        cmd = f"""rm -f {os.path.join(_output_directory, df_name, "*.hea")} -v"""
        print(f"executing --- {cmd}")
        # subprocess.Popen(cmd, shell=True)
        for f in glob(os.path.join(_output_directory, df_name, "*.hea")):
            os.remove(f)
        # copy new headers
        hf_name = hf.replace(".tar.gz", "")
        cmd = f"""cp {os.path.join(headers_tmp, hf_name, "*.hea")} {os.path.join(_output_directory, df_name)} -v"""
        print(f"executing --- {cmd}")
        # subprocess.Popen(cmd, shell=True)
        for f in glob(os.path.join(headers_tmp, hf_name, "*.hea")):
            shutil.copy2(f, os.path.join(_output_directory, df_name))
        print(f"{df_name} done! --- {i+1}/{len(_data_files)}")
        time.sleep(3)
    # remove temporary header files
    shutil.rmtree(headers_tmp)



if __name__ == "__main__":
    # NOT tested yet!!!
    args = get_parser()
    input_directory = args.get("input_directory", BaseCfg.db_dir)
    output_directory = args.get("output_directory", BaseCfg.db_dir)
    run(input_directory, output_directory)
