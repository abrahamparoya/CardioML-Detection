import numpy as np
import argparse
import scipy.signal as sgn
import wfdb
from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
import base64
import json
import os
import wfdb
from preprocess import preprocess_ecg
def read_ecg(path):
    """Read ECG record"""
    return read_wfdb(path)
def read_wfdb(path):
    """Read wfdb record"""
    record = wfdb.rdrecord(path)
    return record.p_signal.T, record.fs, record.sig_name
def main():
    parser = argparse.ArgumentParser(description="Read and preprocess ECG signal")
    parser.add_argument("--exam", default=None, help="Insert path to ECG exam directory, with exam string at the end: /data/isip/data/tnmg_code/v1.0.0/data/S0000000/0000/0000001/TNMG_N1")
    args, unk = parser.parse_known_args()
    path = args.exam
    # read ecg data from wfdb path
    #
    ecg, sample_rate, leads = read_ecg(path)
    processedEcg, new_freq, leads = preprocess_ecg(ecg, sample_rate, leads, new_freq=400, new_len=4096, scale=2, use_all_leads=True)
if __name__ == "__main__":
    main()