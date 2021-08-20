#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Collect blacklisted neurons per parameter per HICANN

Example:
    `python cake_get_calib_statistics.py --wafer 33 --calibs_path ./all_calibs \
                --save_path ./save/blacklisted.h5`
    This collects all blacklisted data in calibs_path and saves a DataFrame as
    hdf5 in ./save/blacklisted.h5
"""





import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

import pyhalco_hicann_v2 as Coordinate
import pylogging
from pycake.helpers.init_logging import init_cake_logging
init_cake_logging([("pycake.collect_statistics", pylogging.LogLevel.DEBUG),
                   ("multiprocessing.MainProcess", pylogging.LogLevel.ERROR)])
logger = pylogging.get('pycake.collect_statistics')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibs_path', '-c', help="path to all calibs", required=True)
    parser.add_argument('--wafer', '-w', help="wafer", type=int, required=True)
    parser.add_argument('--hicanns', nargs='+', type=int,
                        default=list(range(Coordinate.HICANNOnWafer.size)), help="HICANNs to check for")
    parser.add_argument('--save_path', '-s', help="path to save results to", default="blacklisted.h5")

    args = parser.parse_args()
    return args.calibs_path, args.save_path, args.wafer, args.hicanns

def collect_blacklisted(calibs_path, wafer, hicanns):
    blacklisted_dict = defaultdict(dict)
    finished_hicanns = get_all_finished_hicanns(calibs_path, wafer, hicanns)
    for hicann in finished_hicanns:
        search_path = os.path.join(calibs_path, 'w{}_h{}'.format(wafer, hicann))
        if not os.path.isdir(search_path):
            logger.WARN("{}  does not exist.".format(search_path))
            continue
        cal_dirs = [dir_name for dir_name in os.listdir(search_path) if "calibration_f" in dir_name]
        if len(cal_dirs) != 1:
            logger.WARN("Found {} calibration directories. Can not collect data for HICANN {}".format(len(cal_dirs), hicann))
            continue
        for cal_dir in cal_dirs:
            with pd.HDFStore(os.path.join(os.path.join(search_path, cal_dir), 'results.h5')) as store:
                keys_with_blacklisted = [key for key in store if 'defect' in list(store[key].keys())]
                for key in keys_with_blacklisted:
                    blacklisted = np.sum(store[key]['defect'].values)
                    blacklisted_dict[hicann].update({key[1:key.find('_calib')]: blacklisted})
            blacklisted_sum = np.sum(list(blacklisted_dict[hicann].values()))
            logger.INFO("Collected {} blacklisted for HICANN {}".format(blacklisted_sum, hicann))
    blacklisted_df = pd.DataFrame.from_dict(blacklisted_dict, orient='index')
    blacklisted_df.index.names = ['hicann']
    blacklisted_df['wafer'] = wafer
    blacklisted_df.set_index('wafer', append=True, inplace=True)
    return blacklisted_df

def check_report_if_finished(root_path, wafer, hicann):
    report_path = os.path.join(root_path, "report_w{}_h{}.json".format(wafer, hicann))
    with open(report_path, 'rb') as handle:
        finished = json.load(handle).get('calib', {}).get('complete', False)
    return finished

def plot_blacklisted_per_calibration(data_path, wafers=[21,33,37]):
    import pandas as pd
    import numpy as np
    import json
    import matplotlib as mpl
    mpl.use('agg')
    mpl.rcParams.update({"axes.titlesize": 14})
    import matplotlib.pyplot as plt

    dfs = []
    for wafer in wafers:
        with pd.HDFStore(data_path) as store:
            df = store['/blacklisted_w{}'.format(wafer)]
        dfs.append(df)

    df_con = pd.concat([df for df in dfs])
    df_con['index'] = [pp.replace('working', 'calibrated') for pp in df_con['index'].values]
    x_names = np.unique(df_con.sort_values('index')['index'].values)

    fig, ax = plt.subplots(figsize=(8,4.5))
    hatch_styles = ['\\', '.', '//', 'x']
    colors = ['C{}'.format(kk) for kk in range(10)]
    width = 1. / (len(df_con)+1)
    x_range = np.arange(0,len(x_names),1)
    for ii, (wafer, series) in enumerate(df_con.groupby(level='wafer')):
            ax.bar(x_range+(ii-1)*width,
                   series.sort_values('index')['count'].values,
                   width=width/1.02,
                   align='center',
                   alpha=1.,
                   color=colors[ii],
                   label=r'{}'.format(wafer),
                   hatch=hatch_styles[ii])
    ax.legend(loc='upper left', title='wafer')
    ax.set_xticks(x_range)
    ax.set_xticklabels([r"\texttt{{{}}}".format(name) for name in x_names], rotation=10)
    ax.set_ylabel(r"count")
    ax.set_title(r"Calibrated neuron distribution")
    plt.tight_layout()
    plt.savefig("neuron_stats_calibrated.png", dpi=250.)
    plt.close()

def get_all_finished_hicanns(calib_path, wafer, hicanns):
    plot_data_path = os.path.join(calib_path, "plot_data_w{}.json".format(wafer))
    complete_hicanns = []
    with open(plot_data_path, 'rb') as handle:
        plot_data = json.load(handle)['hicann_dict']
    for hicann in hicanns:
        finished = plot_data.get(str(hicann), {}).get('complete_calib', False)
        if finished:
            complete_hicanns.append(hicann)
    return complete_hicanns

def save_as_json(save_path, data_dict):
    if os.path.isfile(save_path):
        os.remove(save_path)
    with open(save_path, 'wb') as json_handle:
        json.dump(data_dict, json_handle, indent=4, separators=(',', ': '))

def save_df(save_path, df):
    wafer = df.index.get_level_values('wafer')[0]
    df.to_hdf(save_path, 'blacklisted_{}'.format(wafer), mode='a', complib='blosc')

def collect_and_save(calibs_path, save_path, wafer, hicanns):
    df = collect_blacklisted(calibs_path, wafer, hicanns)
    save_df(save_path, df)

def main():
    calibs_path, save_path, wafer, hicanns = parse_args()
    collect_and_save(calibs_path, save_path, wafer, hicanns)

if __name__ == "__main__":
    main()

