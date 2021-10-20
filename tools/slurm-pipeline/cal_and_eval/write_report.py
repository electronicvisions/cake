#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" JSON report writer.

Writes a json report file after the 'calib' or 'eval' step of the
cake-slurm-pipeline exited.

Example:
    $ python write_report.py 33 64 "./temp_errors.txt" 1 0 'calib' \
      "./json_test.json"
    writes to ./json_test.json:
        {
        "calib": {
            "0": {
                "complete": false,
                "date": "2017-07-11, 16:17:15",
                "error": ""
            },
            "complete": false,
            "error": ""
        },
        "hicann": 64,
        "wafer": 33
        }
    Further calls to the same file amend this file.
"""

import os
import json
import time
from collections import OrderedDict

def is_complete(exit_code):
    """ Convert bash exit code to finished/not_finished.

    Args:
        exit_code (int): The exit code of the pipeline step

    Returns:
        bool: True if exit code is 0, else False
    """
    if exit_code == 0:
        return True
    else:
        return False

def main(wafer, hicann, error_log, resume_number, exit_code, calib_eval, json_report_path):
    """ Write the JSON file.

    Args:
        wafer (int): Wafer enum
        hicann (int): HICANN enum
        error_log (str): Path to the error logfile of the pipeline step
        resume_number (int): Number of the run
        exit code (int): The exit code of the pipeline step
        calib_eval (str): String to identify if this data belongs to a calib or
                          eval step. May be 'calib' or 'eval'
        json_report_path (str): Path to the json file to write
    """
    complete = is_complete(exit_code)
    if os.path.isfile(error_log):
        with open(error_log, 'r') as log:
            error_message = log.readlines()
    else:
        error_message = ""

    json_dict = {'wafer': wafer, 'hicann': hicann}
    resume_dict = OrderedDict({resume_number: {'complete': complete,
                                   'date': time.strftime("%Y-%m-%d, %H:%M:%S"),
                                   'error': error_message},
                   'error': error_message, 'complete': complete})
    if os.path.isfile(json_report_path):
        with open(json_report_path, 'r') as report:
            json_data = json.load(report)
        if json_data['wafer'] != wafer or json_data['hicann'] != hicann:
            raise ValueError("Wafer or HICANN are not the same as in the previous calibration!")
        json_data.update(json_dict)
    else:
        json_data = json_dict
    if calib_eval in list(json_data.keys()):
        json_data[calib_eval].update(resume_dict)
    else:
        json_data[calib_eval] = resume_dict

    with open(json_report_path, 'w') as report:
        json.dump(json_data, report, indent=4, sort_keys=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save an error message to a json file")
    parser.add_argument("wafer", type=int, help="wafer of this log")
    parser.add_argument("hicann", type=int, help="HICANN of this log")
    parser.add_argument("error_log", help="File that contains the error message")
    parser.add_argument("resume_number", type=int, help="number of this run")
    parser.add_argument("exit_code", type=int, help="1 if the calibration is NOT complete")
    parser.add_argument("calib_eval", help="string which is either 'calib' or 'eval'")
    parser.add_argument("json_report_path", help="path to the json report file")
    args = parser.parse_args()

    if not (args.calib_eval == "calib" or args.calib_eval == "eval"):
        raise ValueError("calib_eval has to be 'calib' or 'eval'")
    main(args.wafer, args.hicann, args.error_log, args.resume_number, args.exit_code,
         args.calib_eval, args.json_report_path)
