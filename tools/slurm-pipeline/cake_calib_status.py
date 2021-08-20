#!/usr/bin/env python

""" Convenience functions to operate on the data produced with cake_calib_hicanns.sbatch

"""





import sys
import os
import json
import shutil
from collections import defaultdict
from distutils.dir_util import copy_tree

def get_report_paths(calibs_dir):
    report_paths = []
    for root, dirs, files in os.walk(calibs_dir, topdown=False):
        for filename in files:
            if filename.startswith("report_")  and filename.endswith('.json'):
                report_paths.append(os.path.join(root, filename))
    return report_paths

def get_num_calibrated_hicanns(calibs_dir):
    report_paths = get_report_paths(calibs_dir)
    completed = dict(calib=0, eval=0)
    for rep_path in report_paths:
        cal_eval_report = load_json(rep_path)
        for cal_eval in ['calib', 'eval']:
            if cal_eval_report.get(cal_eval, {}).get('complete', False):
                completed[cal_eval] += 1
    return completed

def get_calib_status(calibs_dir):
    report_paths = get_report_paths(calibs_dir)
    for rep_path in report_paths:
        cal_eval_report = load_json(rep_path)
        yield cal_eval_report["wafer"], cal_eval_report["hicann"], cal_eval_report.get('calib', {}).get('complete', False)

def copy_complete_calibs(calibs_dir, copy_dir):
    report_paths = get_report_paths(calibs_dir)
    for rep_path in report_paths:
        cal_eval_report = load_json(rep_path)
        if cal_eval_report.get('calib', {}).get('complete', False):
            backends_dir = os.path.join(os.path.split(rep_path)[0], "backends")
            if os.path.isdir(backends_dir):
                print(backends_dir, " --> ", copy_dir)
                copy_tree(backends_dir, copy_dir)

def jsonKeys2str(x):
    if isinstance(x, dict):
        new_dict = {}
        for k,v in list(x.items()):
            try:
                key = int(k)
            except ValueError:
                key = k
            new_dict[key] = v
        return new_dict
    else:
        return x

def load_json(path):
    with open(path, 'rb') as json_log:
        json_dict = json.load(json_log, object_hook=jsonKeys2str)
    return json_dict

def find_max_defects(calibs_path, wafer, hicanns, store_dir='.'):
    import pandas as pd
    import numpy as np
    store_path = os.path.join(store_dir, 'max_defects.json')
    if os.path.isfile(store_path):
        results_dict = load_json(store_path)
        os.remove(store_path)
    else:
        results_dict = {}
    search_paths = [os.path.join(calibs_path, 'w{}_h{}'.format(wafer, hicann)) for hicann in hicanns]
    defects_dict = defaultdict(dict)
    defects_dict.update(results_dict)
    for hicann, search_path in zip(hicanns, search_paths):
        for root, dirs, files in os.walk(search_path, topdown=False):
            for cal_dir in dirs:
                if "calibration_f" in cal_dir:
                    with pd.HDFStore(os.path.join(os.path.join(root, cal_dir), 'results.h5')) as s:
                        for key in s:
                            try:
                                defects = np.sum(s[key]['defect'].values)
                                defects_dict[wafer][hicann].update({key: defects})
                            except:
                                continue
    with open(store_path, 'wb') as json_handle:
        json.dump(defects_dict, json_handle, indent=4)

def main():
    import argparse

    helpstring = """Convenience functions to operate on the data produced with
        cake_calib_hicanns.sbatch. If called directly providing a calibration path,
        the script will print 3 columns: Wafer, Hicann and Status (boolean). The
        status is True for hicanns having a completed calibration status in their
        individual report_w#_h#.json file."""

    parser = argparse.ArgumentParser(description=helpstring)

    parser.add_argument('input', help="Path to calibration results")
    args = parser.parse_args()

    for wafer, hicann, status in get_calib_status(args.input):
        print(wafer, hicann, status)

if __name__ == "__main__":
    main()

