#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the current wafer calibration status as a wafer map.

Creates a bokeh plot of the wafer, containing blacklisted and error messages.

Example:
    `python cake_plot_wafer_status.py ./full_wafer_data 21`
    Creates the file calibration_status_w21.html with data from
    ./full_wafer_data.

    `python cake_plot_wafer_status.py  ./full_wafer_data/ 37 \
        --fpgas_to_exclude $(echo "0,5,2,14,45" | tr "," " ")`
    Creates the file calibration_status_w37.html with data from
    ./full_wafer_data, marking all HICANNs of FPGA 0, 5, 2, 14 and 45 as
    excluded.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import pylogging
from pycake.helpers.init_logging import init_cake_logging
init_cake_logging([("pycake.plot_wafer_status", pylogging.LogLevel.DEBUG),
                   ("multiprocessing.MainProcess", pylogging.LogLevel.ERROR)])
logger = pylogging.get('pycake.plot_wafer_status')

import os
import json
import pandas
import numpy as np
from collections import defaultdict, OrderedDict

from bokeh.plotting import figure, output_file, ColumnDataSource
from bokeh.models.glyphs import Text
from bokeh.resources import CDN, INLINE
from bokeh.embed import file_html
from bokeh.layouts import layout
from bokeh.models import HoverTool
from bokeh.models.widgets import Paragraph

import bokeh.palettes

from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

import pyhalco_hicann_v2 as Coordinate
from pyhalco_common import iter_all, Enum

def get_hicanns_from_dnc(dnc):
    """ Gives formatted list of HICANN enums for given dnc enum

    Args:
        dnc (int): DNC enum

    Returns:
        numpy.ndarray: HICANN enums
    """
    per_dnc = Coordinate.HICANNOnDNC.size
    offset = (per_dnc // 2)
    dnc = Coordinate.DNCOnWafer(Enum(dnc))
    h0 = Coordinate.HICANNOnDNC(Enum(0)).toHICANNOnWafer(dnc).toEnum().value()
    h1 = Coordinate.HICANNOnDNC(Enum(per_dnc // 2)).toHICANNOnWafer(dnc).toEnum().value()
    return np.array([range(h0, h0+offset) , range(h1, h1+offset)])

def get_hicanns_from_fpga(fpga, wafer):
    """ Gives formatted list of HICANN enums for given fpga enum

    Args:
        fpga (int): FPGA enum

    Returns:
        numpy.ndarray: HICANN enums
    """
    return  np.array([hc_glob.toHICANNOnWafer().toEnum().value()
             for hc_glob in Coordinate.FPGAGlobal(Coordinate.FPGAOnWafer(Enum(fpga)),
                                     Coordinate.Wafer(wafer)).toHICANNGlobal()])

def get_dnc_from_hicann(hicann):
    """ Gives DNC enum for given hicann enum

    Args:
        hicann (int): hicann enum

    Returns:
        int: DNC enum
    """
    return Coordinate.HICANNOnWafer(Enum(hicann)).toDNCOnWafer().toEnum().value()

def get_fpga_from_hicann(hicann):
    """ Gives FPGA enum for given hicann enum

    Args:
        hicann (int): hicann enum

    Returns:
        int: FPGA enum
    """
    # wafer enum should not matter as long as it is > 3 ==> set to 33
    wafer = 33
    return (Coordinate.HICANNGlobal(Coordinate.HICANNOnWafer(Enum(hicann)),
            Coordinate.Wafer(Enum(wafer))).toFPGAOnWafer().value())

def get_color_blacklisted(num_bl, cmap_dict):
    """ Get hex color depending on num_bl

    Args:
        num_bl (int): number of blacklisted neurons of the HICANN
        cmap_dict (dict): of the form {num_blacklisted: hex color value

    Returns:
        str: hex color code
    """
    return cmap_dict[num_bl]

def sort_error_dict(error_dict):
    """ sorts the given dict depending on the error count

    Args:
        error_dict (dict): A dict containing errors (key) and the respective
                           count (value)

    Returns:
        OrderedDict: keys are error messages, values are error count (sorted)
    """
    if not error_dict:
        return OrderedDict()
    error_list = [[]]*len(error_dict.keys())
    keys, values = zip(*[[v,k] for v,k in error_dict.iteritems()])
    for ii,position in enumerate(np.argsort(values)):
        error_list[ii] = [keys[position], values[position]]
    ordered_dict = OrderedDict(error_list[::-1])
    return ordered_dict

def make_color_map_errors(error_dict):
    """ Creates a color map for the error plot

    Args:
        error_dict (dict): A dict containing errors (key) and the respective
                           count (value)

    Returns:
        dict: keys are error message, values are hex color codes
    """
    ordered_dict = sort_error_dict(error_dict)
    num_different_errors = len(ordered_dict.keys())
    colors_list = []
    for key in ['Spectral', 'RdGy', 'Set2', 'Paired', 'Set3', 'RdYlBu', 'RdPu', 'PiYG']:
        palette = bokeh.palettes.small_palettes[key]
        colors_list += palette[max(palette.keys())]
    colors_list_unique = []
    for color in colors_list:
        if color not in colors_list_unique:
            colors_list_unique.append(color)
    colors = {key: colors_list_unique[ii % len(colors_list_unique)]
              for ii, key in enumerate(ordered_dict.keys())}
    colors[''] = '#00fff2'
    colors['excluded'] = '#2b2c2d'
    return colors

def get_wafer_data():
    """ Contains data about the wafer geometry

    Returns:
        int: HICANNs per reticle
        int: number of rows of HICANNs on a reticle
        int: width of a reticle
        int: height of a HICANN
        numpy.ndarray: 2D array containing DNC number for given (x,y)
                       coordinate
    """
    hicanns_per_reticle = 8
    rows = 2
    ret_width = hicanns_per_reticle // rows
    hc_height = ret_width // rows

    dnc2reticle_map = np.array([
    [-1, -1, -1,  0,  1,  2, -1, -1, -1],
    [-1, -1,  3,  4,  5,  6,  7, -1, -1],
    [-1,  8,  9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31, 32],
    [-1, 33, 34, 35, 36, 37, 38, 39, -1],
    [-1, -1, 40, 41, 42, 43, 44, -1, -1],
    [-1, -1, -1, 45, 46, 47, -1, -1, -1] ])
    return hicanns_per_reticle, rows, ret_width, hc_height, dnc2reticle_map

def save_errors_as_latex(hicann_dict, wafer, path=None):
    if path is None:
        path = 'calibration_status_error_w{}_{}.tex'
    for cal_eval in ['calib', 'eval']:
        save_path = path.format(wafer, cal_eval)
        _, error_dict = get_plot_data(hicann_dict, cal_eval=cal_eval)
        tex_lines = get_errors_as_latex(error_dict, cal_eval, wafer)
        with open(save_path, 'wb') as handle:
            handle.write("\n".join(tex_lines))

def get_errors_as_latex(error_dict, cal_eval, wafer):
    tex_escape_chars = ['&', '%', '$', '#', '_', '{', '}', '~', '^']

    ordered_dict = sort_error_dict(error_dict)
    colors = make_color_map_errors(error_dict)
    color_def = {}
    color_name = "errorcolorw{0}c{1}{{}}".format(wafer, cal_eval)
    for ii, error in enumerate(ordered_dict.iterkeys()):
        c_name = color_name.format(ii)
        c_def = r"\definecolor{{{1}}}{{HTML}}{{{0}}}".format(colors[error].replace("#", ""), c_name)
        color_def[error] = (c_name, c_def)

    tex_lines = [c_def[1] for c_def in color_def.itervalues()]
    if cal_eval == "calib":
        cal_eval_long = "calibration"
    else:
        cal_eval_long = "evaluation"
    tex_head = r"\textbf{{Errors during {0} of wafer {1}:}}".format(cal_eval_long, wafer)
    tex_lines.append(tex_head)
    tex_begin = r"\begin{description}[labelsep=1mm, align=left, itemsep=-6pt]"
    tex_lines.append(tex_begin)
    for error, num_errors in ordered_dict.iteritems():
        error_esc = error
        for esc_char in tex_escape_chars:
            error_esc = error_esc.replace(esc_char, "\{}".format(esc_char))
        tex_item = r"\item [\textcolor{{{0}}}{{\rule{{20pt}}{{8pt}}}}] {1}x {2}".format(color_def[error][0], num_errors, error_esc)
        tex_item = tex_item.replace("\n", "")
        tex_lines.append(tex_item)
    tex_end = r"\end{description}"
    tex_lines.append(tex_end)
    return tex_lines

def get_plot_data(hicann_dict, color_code='blacklisted', cal_eval='calib'):
    """ Accumulates plot data such as number of blacklisted etc.

    Args:
        hicann_dict (dict): A dict containing information about each HICANN
                            such as errors, blacklisted, etc.
        color_code (str): Defines if the color code for blacklisted or errors
                          should be used. Can be 'blacklisted' or 'error'.
        cal_eval (str): if the plot is of calib or eval data. Can be 'calib'
                        or 'eval'

    Returns:
        dict: Data for the plot, keys are plotting arguments (hicann,
              error_calib, ...), values are a list
        dict: Error count, keys are error messages, values are error count
    """
    hicanns_per_reticle, rows, ret_width, hc_height, dnc2reticle_map = get_wafer_data()

    dncs = [dnc for row in dnc2reticle_map for dnc in row if dnc != -1]
    hicann_array = np.array([get_hicanns_from_dnc(dnc) for dnc in dncs])
    ret_coords = [(ret_width*j,ret_width*i) for i,row in enumerate(dnc2reticle_map)
                                            for j,dnc in enumerate(row) if dnc != -1]
    hc_coords =  [[hicann_array[i,j,k], x+pos, y+hc_height*row]
                  for i, (x,y) in enumerate(ret_coords)
                  for j,row in enumerate(range(rows))
                  for k,pos in enumerate(range(ret_width))]
    hc_xy = np.array(zip(*hc_coords))

    plot_data = defaultdict(list)
    error_dict = defaultdict(int)
    plot_data.update(dict(x=hc_xy[1], y=-hc_xy[2], hicann=hc_xy[0],
                          DNC=[get_dnc_from_hicann(hc) for hc in hc_xy[0]],
                          FPGA=[get_fpga_from_hicann(hc) for hc in hc_xy[0]]))

    unique_keys = np.unique([k for value in hicann_dict.itervalues() for k in value.keys()])
    for hc in hc_xy[0]:
        for key in unique_keys:
            if key in ["error_calib", "error_eval"]:
                continue
            else:
                plot_data[key].append(hicann_dict.get(hc, {}).get(key, None))
        for key in ["error_calib", "error_eval"]:
            if hicann_dict[hc]['blacklisted'] == -2: #HC is excluded, so we do not care for the error
                error_dict['excluded'] += 1
                lines = ['excluded']
            else:
                lines = []
                for line in hicann_dict.get(hc, {}).get(key, []):
                    # FIXME: better filter for spurious errors
                    if "Broken pipe" in line:
                        continue
                    if any([err_str in line for err_str in get_line_check_strings()]):
                        lines.append(line)
                        err_key = get_error_key(line)
                        if cal_eval in key:
                            error_dict[err_key] += 1
                        break
            plot_data[key].append("".join(lines))
    plot_data = set_fill_color(plot_data, error_dict, color_code, cal_eval)
    return plot_data, error_dict

def get_error_key(line):
    error_keys_unique_dict = get_error_keys_with_unique_numbers()
    for err_unique, replace_str in error_keys_unique_dict.items():
        if err_unique in line:
            return replace_str
    return line

def get_line_check_strings():
    """We check each line for multiple error strings like "Error", "what()" etc."""
    return ["Error", "slurmstepd: error:", "recvfrom (err=11)",
            "No reset response from FPGA!", "what():", "error_already_set",
            "without an active exception", ]

def get_error_keys_with_unique_numbers():
    """ Some errors have long error messages or numbers like HICANN(XX) or IP
        192... in them. We want to generalize
    """
    error_keys_unique_dict = {
            "ound ADC frequency of": ("RuntimeError: Measured ZZZZ spikes in the HICANN preout "
                                      "and found ADC frequency of XXXX MHz, this is unlikely "
                                      "(expected ~YYYY MHz). Is L1 OK?"),
            "Dangerous HICANN configuration": ("Dangerous HICANN configuration of "
                                               " HICANNOnWafer(Enum(XXXX))abort configuration!"),
            "[34: Client connect operation failed. OS: 111": ("RuntimeError: [34: Client connect "
                                                              "operation failed. OS: 111 - Connection "
                                                              "refused][1: Operating system][111: "
                                                              "[.......]"),
            "No HICANN config packet received": ("HostALController::getReceivedHICANNConfig() [XX:X]: "
                                                 "No HICANN config packet received"),
            "cannot open connection to Software": ("XXX.XXX.XX.XXX: cannot open connection to Software ARQ daemon"),
            "Server-side user exception": ("RuntimeError: [Remote][6: Server-side user exception. Exception type: "
                                           "St13runtime_error. Exception message: wrong boardId.][0: No sub system.][0: "
                                           "][What: wrong boardId][Context: ../lib-rcf/src/RCF/Exception.cpp(482): virtual "
                                           "void RCF::RemoteException::throwSelf() const: : Thread-id=XXXX :"
                                           "Timestamp(ms)=YYY: : ]"),
            "CANCELLED AT ": ("slurmstepd: error: *** JOB XXXXX ON HostYYY CANCELLED AT TIMEZZZ ***"),
            "No reset response from FPGA!" : ("No reset response from FPGA!")
    }
    return error_keys_unique_dict

def get_cmap_dict(max_bl=Coordinate.NeuronOnHICANN.size, cmap_name='rainbow'):
    cmap_mpl = get_cmap('rainbow')
    cmap = [cmap_mpl(int(ii*(1.*cmap_mpl.N/(max_bl+1)))) for ii in range(max_bl+1)]
    cmap_hex = [rgb2hex(rgba[:3]) for rgba in cmap]
    cmap_dict = {ii: color for ii, color in enumerate(cmap_hex)}
    cmap_dict[-1] = '#3d3d29' # HICANN on excluded FPGA
    cmap_dict[-2] = '#2b2c2d' # calibration not finished
    return cmap_dict

def set_fill_color(plot_data, error_dict, color_code='blacklisted', cal_eval='calib'):
    """ Determines fill color for each HICANN

    Args:
        plot_data (dict): Data for the plot, keys are plotting arguments (hicann,
              error_calib, ...), values are a list
        error_dict (dict): Error count, keys are error messages, values are error count
        color_code (str): Defines if the color code for blacklisted or errors
                          should be used. Can be 'blacklisted' or 'error'.
        cal_eval (str): if the plot is of calib or eval data. Can be 'calib'
                        or 'eval'

    Returns:
        dict: plot_data, appended by a list of fill colors
    """
    if color_code =='blacklisted':
        cmap_dict = get_cmap_dict(max_bl=Coordinate.NeuronOnHICANN.size, cmap_name='rainbow')
        for blacklisted in plot_data['blacklisted']:
            plot_data['fill_color'].append(get_color_blacklisted(blacklisted, cmap_dict))
    elif color_code == 'error':
        hc_colors = ['#000000']*len(plot_data['hicann'])
        colors = make_color_map_errors(error_dict)
        for ii,complete in enumerate(plot_data['complete_{}'.format(cal_eval)]):
            if complete:
                hc_colors[ii] = '#339933'
            else:
                if plot_data['blacklisted'][ii] == -2:
                    err_key = 'excluded'
                else:
                    err_key = get_error_key(plot_data['error_{}'.format(cal_eval)][ii])
                hc_colors[ii] = colors[err_key]
        plot_data['fill_color'] = hc_colors
    return plot_data

def get_pie_chart_colors(idx):
    colors_dict = {'blacklisted': '#FF0000',
                   'not blacklisted': '#009933',
                   'not yet calibrated': '#3399ff',
                   'excluded': '#2b2c2d'
                  }
    for key, color in colors_dict.items():
        if key in idx:
            return color
    return '#000000'

def get_blacklisted_series(hicann_dict):
    """ Returns a DataFrame for the pie chart

    Args:
        hicann_dict (dict): A dict containing information about each HICANN
                            such as errors, blacklisted, etc.
    """
    num_neurons_on_hicann = Coordinate.NeuronOnHICANN.size
    num_hicanns_on_wafer = Coordinate.HICANNOnWafer.size
    total_num_neurons = num_neurons_on_hicann * num_hicanns_on_wafer
    blacklisted_series = pandas.Series([0,0,0,0],
                                        index=['blacklisted', 'not blacklisted', 'excluded', 'not yet calibrated'])
    for hicann, data in hicann_dict.iteritems():
        if data['blacklisted'] > -1:
            blacklisted_series['blacklisted'] += data['blacklisted']
            blacklisted_series['not blacklisted'] += num_neurons_on_hicann - data['blacklisted']
        elif data['blacklisted'] == -1:
            blacklisted_series['not yet calibrated'] += num_neurons_on_hicann
        elif data['blacklisted'] == -2:
            blacklisted_series['excluded'] += num_neurons_on_hicann
    total_measured_neurons = blacklisted_series['blacklisted'] + blacklisted_series['not blacklisted']
    new_idx = []
    for key in blacklisted_series.index:
        if total_num_neurons != 0:
            percentage_total = 100.*blacklisted_series[key]/total_num_neurons
        else:
            percentage_total = np.inf
        if total_measured_neurons != 0:
            percentage_measured = 100.*blacklisted_series[key]/total_measured_neurons
        else:
            percentage_measured = np.inf
        new_idx.append(r"{} {}, {:.1f} %".format(blacklisted_series[key], key,
                       percentage_total))
        if key in ['not blacklisted', 'blacklisted']:
            new_idx[-1] += ", ({:.1f} %)".format(percentage_measured)
    blacklisted_series.index = pandas.Index(new_idx)
    # set correct colors, this is necessary as dict is not Ordered
    df = pandas.DataFrame(blacklisted_series)
    df.reset_index(inplace=True)
    df.columns = ['index', 'count']
    df['color'] = df['index'].apply(get_pie_chart_colors)
    df.sort_values('index', inplace=True)
    return df, blacklisted_series

def pie_chart_blacklisted_matplotlib(hicann_dict, wafer):
    """ Creates a matplotlib pie chart of the blacklisted.

    Args:
        hicann_dict (dict): A dict containing information about each HICANN
                            such as errors, blacklisted, etc.
        wafer (int): wafer
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    blacklisted_df, _ = get_blacklisted_series(hicann_dict)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = blacklisted_df['index'].values
    sizes =  blacklisted_df['count'].values
    explode = [0]*len(sizes)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, labeldistance=.3, radius=9.,
                    shadow=False, startangle=0, colors=list(blacklisted_df['color'].values))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('pie_chart_w{}.png'.format(wafer), bbox_inches='tight')
    return fig, ax

def pie_chart_blacklisted(hicann_dict):
    """ Creates a bokeh pie chart of the blacklisted.

    Args:
        hicann_dict (dict): A dict containing information about each HICANN
                            such as errors, blacklisted, etc.
    """
    blacklisted_df, ss = get_blacklisted_series(hicann_dict)
    bokeh_figure = Donut(data=ss, plot_width=800, plot_height=800,
                         color=list(blacklisted_df['color'].values), title="neuron partitioning",
                         text_font_size='12pt')
    bokeh_figure.background_fill_alpha = 0.
    bokeh_figure.border_fill_alpha = 0.
    return bokeh_figure

def plot_wafer(hicann_dict, wafer, color_code='blacklisted', cal_eval='calib'):
    """ Creates the bokeh figure of a wafer plot

    Args:
        hicann_dict (dict): A dict containing information about each HICANN
                            such as errors, blacklisted, etc.
        wafer (int): Wafer enum
        color_code (str): Defines if the color code for blacklisted or errors
                          should be used. Can be 'blacklisted' or 'error'.
        cal_eval (str): if the plot is of calib or eval data. Can be 'calib'
                        or 'eval'

    """
    hc_height = get_wafer_data()[3]
    unique_keys = np.unique([k for value in hicann_dict.itervalues() for k in value.keys()])

    plot_data, error_dict = get_plot_data(hicann_dict, color_code, cal_eval)
    source = ColumnDataSource(data=plot_data)

    tooltip_keys = plot_data.keys()
    for key in ['x', 'y', 'fill_color']:
        tooltip_keys.remove(key)
    tooltips = [(key, "@{}".format(key)) for key in sorted(tooltip_keys)]
    hover = HoverTool(tooltips=tooltips)
    if color_code == 'blacklisted':
        title = "{}s of wafer {}".format(color_code, wafer)
    else:
        title = "{}s of wafer {} -- {}".format(color_code, wafer, cal_eval)
    bokeh_figure = figure(plot_width=800, plot_height=800, tools=[hover],
               title=title)
    bokeh_figure.xgrid.grid_line_color = None
    bokeh_figure.ygrid.grid_line_color = None
    bokeh_figure.axis.visible = False
    bokeh_figure.background_fill_alpha = 0.
    bokeh_figure.border_fill_alpha = 0.

    bokeh_figure.rect('x', 'y', height=hc_height, width=1, fill_color='fill_color',
                      source=ColumnDataSource(data=plot_data), line_color='black')
    if color_code == 'blacklisted':
        add_text(bokeh_figure, plot_data)

    # This does not exist in version 0.12.4, but will be useful in later versions
    #save_svg = True
    #if save_svg:
    #    from bokeh.io import export_svgs
    #    bokeh_figure.output_backend = "svg"
    #    export_svgs(bokeh_figure, filename="wafer_plot_{}_{}_{}.svg".format(wafer, color_code, cal_eval))
    return bokeh_figure

def plot_cmap(max_bl=512, scale_factor=0.4, cmap_name='rainbow', use_light_colors=True):
    import matplotlib as mpl
    mpl.use('agg')
    mpl.rcParams.update({"axes.titlesize": 14})
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
    import matplotlib.ticker as ticker

    cmap_mpl = get_cmap(cmap_name)
    cmap = []
    for ii in range(max_bl):
        col = np.array(cmap_mpl(int(ii*(1.*cmap_mpl.N/(max_bl)))))
        if use_light_colors:
            scale_factor = 0.4
            col = col + np.array([scale_factor,scale_factor,scale_factor,0])
            col[col>1] = 1
        cmap.append(col)
    cmap_lin = LinearSegmentedColormap.from_list('newmap', cmap)
    fig, ax = plt.subplots(figsize=(1,3))
    bounds = list(range(max_bl+1))
    norm = BoundaryNorm(bounds, cmap_lin.N)
    ticks = np.linspace(0, max_bl, 8, dtype=int)
    cb2 = ColorbarBase(ax, cmap=cmap_lin,
                                    norm=norm,
                                    # to use 'extend', you must
                                    # specify two extra boundaries:
                                    boundaries=bounds,
                                    #extend='both',
                                    ticks=ticks,  # optional
                                    spacing='uniform',
                                    orientation='vertical')
    plt.tight_layout()
    plt.savefig("cmap_blacklisted.png")
    plt.close()

def add_text(figure, plot_data):
    """ Add the blacklisted as text over the wafer plot

    Args:
        figure (bokeh figure): The bokeh figure of the wafer
        plot_data (dict): Data for the plot, keys are plotting arguments (hicann,
              error_calib, ...), values are a list
    """
    text_source = defaultdict(list)
    text_source.update(dict(x=np.add(plot_data['x'], 0.5), y=plot_data['y'],
                            text=plot_data['blacklisted']))
    source = ColumnDataSource(text_source)
    glyph = Text(x="x", y="y", text="text", angle=1.57, text_color="#000000", text_align='center')
    figure.add_glyph(source, glyph)

def get_blacklisted(wafer, hicann, backend_path):
    """ Get blacklisted of the redman xml in backend_path

    Args:
        wafer (int): Wafer enum
        hicann (int): HICANN enum
        backend_path (str): Path to the redman xml backend

    Returns:
        dict: keys are HICANN enums, values are number of blacklisted
    """
    import pyredman as redman
    import pycake.helpers.redman

    wafer = Coordinate.Wafer(Enum(wafer))
    hicann = Coordinate.HICANNOnWafer(Enum(hicann))
    try:
        redman = pycake.helpers.redman.Redman(
            backend_path, Coordinate.HICANNGlobal(hicann,wafer))
        neuron_enums = range(Coordinate.NeuronOnHICANN.size)
        blacklisted = {ii: not redman.hicann_with_backend.neurons().has(Coordinate.NeuronOnHICANN(Enum(ii)))
                   for ii in neuron_enums}
        return blacklisted
    except RuntimeError:
        logger.ERROR("Could not load blacklisted for wafer {}, HICANN {}".format(wafer, hicann))
        return {}

def collect_blacklisted(calibs_path, wafer, hicanns):
    """
    Args:
        calibs_path: path to all calibs
        wafer (int): wafer to collect blacklisted from
        hicanns (list of int): list of hicanns to get blacklisted from

    Returns:
        dict: keys are HICANN enums, values are number of blacklisted
    """
    hicann_dict = defaultdict(dict)
    for root, dirs, files in os.walk(calibs_path, topdown=False):
        for filename in files:
            if filename.endswith(").xml"):
                hicann = int(filename[filename.index('Enum(')+5:filename.index(').xml')])
                wafer_from_filename = int(filename[filename.index('Wafer(')+6:filename.index(')')])
                if hicann in hicanns and wafer == wafer_from_filename:
                    blacklisted = get_blacklisted(wafer, hicann, root)
                    num_bl = np.sum(blacklisted.values()) if blacklisted else -1
                    if hicann in hicann_dict.keys():
                        logger.WARN("Found more than one calibration file "
                              "for the same wafer, HICANN: {}, {}!!".format(wafer, hicann))
                    hicann_dict[hicann]['blacklisted'] = num_bl
    not_found_hicanns = [h for h in hicanns if h not in hicann_dict.keys()]
    if not_found_hicanns != []:
        logger.WARN("For the following HICANNs, no redman file "
              "could be found: {}".format(not_found_hicanns))
    return hicann_dict

def get_reports(calibs_path):
    """
    Args:
        calibs_path: Path to all calibs

    Returns:
        list: List of logs which contain wafer, hicann, errors, complete
    """
    logs = []
    for root, dirs, files in os.walk(calibs_path, topdown=False):
        for filename in files:
            if "report_" in filename and filename.endswith('.json'):
                with open(os.path.join(root, filename), 'rb') as json_log:
                    logger.INFO("Fetched log from {}".format(os.path.join(root, filename)))
                    log = json.load(json_log)
                    logs.append(log)
    return logs

def get_finished_calibs(logs, wafer):
    """
    Args:
        logs (list): List of json logs

    Returns:
        defaultdict: Wafer_enum (int): hicann_enums (list of int)
    """
    completed_hicanns = {"calib": [], "eval": []}
    for log in logs:
        if log['wafer'] != wafer:
            continue
        hicann = log['hicann']
        for key in ['calib', 'eval']:
            if log.get(key, {}).get('complete', False):
                if hicann in completed_hicanns[key]:
                    logger.WARN("HICANN {} is listed more than once...is something wrong "
                            "with the detection of finished {}s?".format(hicann, key))
                else:
                    completed_hicanns[key].append(hicann)
    return completed_hicanns

def sort_hicann_logs(logs, wafer):
    """ Convert list of logs to dict like {hicann_enum: dict with error information}

    Args:
        logs (list): List of json logs
        wafer (int): Wafer enum
    """
    hicann_dict = defaultdict(dict)
    for log in logs:
        if log['wafer'] != wafer:
            continue
        hicann = log['hicann']
        for key in ['calib', 'eval']:
            for log_key, log_value in log.get(key, {}).iteritems():
                if not isinstance(log_value, dict):
                    hicann_dict[hicann].update({'{}_{}'.format(log_key, key): log_value})
    return hicann_dict

def amend_errors_html(html, error_dict, headline):
    """ Amend the given html str with the error color code

    Args:
        html (str): HTML code generated by bokeh
        error_dict (dict): Error count, keys are error messages, values are error count
        headline (str): Str to be written before the color coded errors

    Returns:
        str: html amended by color: error legend
    """
    new_html = html[:html.find('</body>\n</html>')]
    colors = make_color_map_errors(error_dict)
    ordered_dict = sort_error_dict(error_dict)
    if "Calib" in headline:
        for _ in range(40):
            new_html += "\n<br>"
    new_html += "\n<p>" + headline + "<br>\n"
    for key, value in ordered_dict.iteritems():
        new_html += ("""<span style="background-color:{};"> _______</span>""".format(colors[key])
                         + "\n{}x: {} <br>".format(value, key))
    new_html += '</p>\n</body>\n</html>'
    return new_html

def save_plot_data(save_dict, save_path):
    # convert numpy arrays to lists to save it as json
    for key, value in save_dict.iteritems():
        for key2, value2 in value.iteritems():
            if isinstance(value2, np.ndarray):
                save_dict[key][key2] = list(value2)
    with open(save_path, 'wb') as handle:
        json.dump(save_dict, handle, indent=4, separators=(',', ': '))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("calibs_path", help="path to all calibs")
    parser.add_argument("--wafer", type=int, help="wafer to plot", nargs="+", required=True)
    parser.add_argument("--fpgas_to_exclude", type=int, nargs='+',
                        help="to mark specific FPGAs as down/blacklisted")
    parser.add_argument("--load_path", type=str,
                        help="load data from this path instead of reports")

    args = parser.parse_args()
    return args.calibs_path, args.wafer, args.fpgas_to_exclude, args.load_path

def get_blacklisted_hicanns(hicann_dict, error_code=-1):
    hcs = [hc for hc, dic in hicann_dict.items() if dic['blacklisted'] == error_code]
    return hcs

def exclude_fpgas(hicann_dict, fpgas_to_exclude, wafer):
    for fpga in fpgas_to_exclude:
        for hc in get_hicanns_from_fpga(fpga, wafer):
            if hicann_dict[hc]['blacklisted'] >= 0:
                logger.WARN("HICANN {}, FPGA {} is marked as excluded, although a calibration value exists".format(hc, fpga))
            else:
                hicann_dict[hc]['blacklisted'] = -2
    return hicann_dict

def set_uncalibrated_hicanns(hicann_dict, calibrated_hicanns, error_code=-1):
    not_calibrated_hicanns = [hc for hc in all_hicanns if hc not in calibrated_hicanns]
    for hc in not_calibrated_hicanns:
        hicann_dict[hc]['blacklisted'] = -1
    return hicann_dict

def jsonKeys2str(x):
    """ Helper function to convert str keys in json to int """
    if isinstance(x, dict):
        dict2 = {}
        for k,v in x.iteritems():
            try:
                key = int(k)
            except ValueError:
                key = k
            dict2[key] = v
        return dict2
    return x

def add_error_messages_html(hicann_dict, html):
    for cal_eval in ['calib', 'eval']:
        title = "Calibration" if cal_eval == 'calib' else "Evaluation"
        _, error_dict = get_plot_data(hicann_dict, cal_eval=cal_eval)
        html = amend_errors_html(html, error_dict, headline=title + " Errors")
    return html

def init_hicann_dict(error_code=-1):
    hicann_dict = defaultdict(dict)
    for hc in [hicann.toEnum().value() for hicann in iter_all(Coordinate.HICANNOnWafer)]:
        hicann_dict[hc]['blacklisted'] = error_code
    return hicann_dict

def get_uncalibrated_hicanns(hicann_dict):
    hcs = []
    for hc, dic in hicann_dict.items():
        if dic['blacklisted'] < 0:
            hcs.append(hc)
    return hcs

def do_wafer_plots(wafer, logs, calibs_path, fpgas_to_exclude):

    logger.INFO("Plotting wafer {} status".format(wafer))

    hicann_dict = init_hicann_dict()
    for hc, log in sort_hicann_logs(logs, wafer).iteritems():
        hicann_dict[hc].update(log)
    calibrated_hicanns = get_finished_calibs(logs, wafer)
    blacklisted_dict = collect_blacklisted(calibs_path, wafer, calibrated_hicanns['calib'])
    for hicann, results in blacklisted_dict.iteritems():
        hicann_dict[hicann].update(results)

    save_dict = {'hicann_dict': hicann_dict}
    save_path = os.path.join(calibs_path, 'plot_data_w{}.json'.format(wafer))

    save_plot_data(save_dict, save_path)

    if fpgas_to_exclude is not None:
        hicann_dict = exclude_fpgas(hicann_dict, fpgas_to_exclude, wafer)

    figs = []
    figs.append(plot_wafer(hicann_dict, wafer, color_code='blacklisted', cal_eval='calib'))
    for cal_eval in ['calib', 'eval']:
        figs.append(plot_wafer(hicann_dict, wafer, color_code='error', cal_eval=cal_eval))

    #pie_chart = pie_chart_blacklisted(hicann_dict)

    filepath = 'calibration_status_w{}.html'.format(wafer)
    output_file(filepath)
    # wafer_plots = layout([[figs[0], pie_chart], [figs[1], figs[2]]])
    wafer_plots = layout([[figs[0]], [figs[1], figs[2]]])

    html = file_html(wafer_plots, INLINE, "Calib Status wafer {}".format(wafer))

    html = add_error_messages_html(hicann_dict, html)
    with open(filepath, 'wb') as html_file:
        html_file.write(html)

    #pie_chart_blacklisted_matplotlib(hicann_dict, wafer)
    #save_errors_as_latex(hicann_dict, wafer, path=None)

def main():
    calibs_path, wafers, fpgas_to_exclude, load_path = parse_args()

    logs = get_reports(calibs_path)

    for wafer in wafers:
        do_wafer_plots(wafer, logs, calibs_path, fpgas_to_exclude)

if __name__ == "__main__":
    main()

