#!/usr/bin/env python
import pandas as pd
# deactivate warning
# http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.lines as mlines
from matplotlib import rcParams
from cycler import cycler
rcParams.update({'figure.autolayout': True, 'axes.formatter.limits': [-4, 4],
                 'figure.figsize': [6.4, 4.8], 'axes.prop_cycle': cycler('color',
                    [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#9467bd',
                     u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])})

# ===========================HOW TO USE (as a plotting script)=============================

# if only calib should be plotted: make_plots.py STORAGE_PATH --cal "CALIBRATION_FOLDER"

# if only eval should be plotted: make_plots.py STORAGE_PATH --eval
# "EVAL_FOLDER" --wafer WAFER_ENUM --hicann HICANN_ENUM --backend_path BACKEND_PATH
# (if you want to deselect defect neurons. The information is only stored in
# the calib DataFrames or the redman xml and to load the xml you need the wafer
# and hicann enums.) If you ommit --wafer etc., plots including defects are still done

# if both should be plotted: make_plots.py STORAGE_PATH --cal "CALIBRATION_FOLDER" --eval
# "EVAL_FOLDER"

# ===========================plot parameters ===============================================
neuron_enums = [0] # neurons to be plotted (trace), if not specified in args.neuron_enum
plot_config = {'readout_shift': {'result_key': 'coeff0', 'result_label': r"offset [V]",
                                 'xunit': 'V', 'yunit': 'V'},
               'V_reset':   {'result_key': 'baseline', 'result_label': r"$V_{reset}$ [V]",
                             'key_label_cal': r"$V_{reset}$ [DAC] (in)", 'xunit': 'V',
                             'per_fg_block': True},
               'V_t':       {'result_key': 'max', 'result_label': r"$V_{t}$ [V]",
                             'key_label_cal': r"$V_{t}$ [DAC] (in)", 'xunit': 'V',
                            },
               'E_synx':    {'result_key': 'mean', 'result_label': r"$E_{syn,x}$ [V]",
                             'key_label_cal': r"$E_{syn,x}$ [DAC] (in)", 'xunit': 'V'},
               'E_syni':    {'result_key': 'mean', 'result_label': r"$E_{syn,i}$ [V]",
                             'key_label_cal': r"$E_{syn,i}$ [DAC] (in)", 'xunit': 'V'},
               'E_l':       {'result_key': 'mean', 'result_label': r"$E_{l}$ [V]",
                             'key_label_cal': r"$E_{l}$ [DAC] (in)", 'xunit': 'V'},
               'I_pl':      {'result_key': 'tau_ref', 'result_label': r"$\tau_{ref}$ [s]",
                             'key_label_cal': r"$I_{pl}$ [DAC]", 'xunit': 's', 'rep': 0,
                             'num_steps': 6, 'logx': True, 'bins': 100,
                             'xrange': np.logspace(-10, -4.5, 200)},
               'I_gl':      {'result_key': 'tau_2', 'result_label': r"$\tau_{m}$ [s]",
                             'key_label_cal': r"$I_{gl}$ [DAC]", 'xunit': 's', 'logx': True,
                             'xrange': np.logspace(-7, -4.5, 200)},
               'V_convoffi': {'result_key': 'coeff0', 'result_label': r"$V_{convoff}$ [DAC]",
                              'xunit': 's', 'DACrange': [0, 1023]},
               'V_convoffx': {'result_key': 'coeff0', 'result_label': r"$V_{convoff}$ [DAC]",
                              'xunit': 's', 'DACrange': [0, 1023]},
               'V_syntci':  {'result_key': 'tau_syn', 'result_label': r"$\tau_{syn,i}$ [s]",
                             'key_label_cal': r"$V_{syntc,i}$ [DAC] (in)", 'xunit': 's',
                             'xrange': np.logspace(-9, -3.5, 200), 'logx': True},
               'V_syntcx':  {'result_key': 'tau_syn', 'result_label': r"$\tau_{syn,x}$ [s]",
                             'key_label_cal': r"$V_{syntc,x}$ [DAC] (in)", 'xunit': 's',
                             'xrange': np.logspace(-9, -3.5, 200), 'logx': True}
              }
for plot_dict in plot_config.itervalues():
    if plot_dict.get('result_label', False):
        plot_dict['key_label_eval'] = plot_dict['result_label'] + r" (in)"

for config in plot_config.itervalues():
    config['nrns'] = neuron_enums
# ==========================================================================================

def make_fig_dir(path, fig_dir):
    """
    creates directory for figures if it is not already there
    """
    fig_path = os.path.join(path, fig_dir)
    if fig_dir not in os.listdir(path):
        os.mkdir(fig_path)
    return fig_path

def load_data(storage_path, calib_name='', eval_name='', patch=True):
    """
    load data from calibration and/or eval run

    Args:
        storage_path [str]: path where calibration and/or evaluation folder are
        calib_name [str]: name of the calibration folder
        eval_name [str]: name of the evaluation folder
        patch [bool]: if True, calculates which neurons are marked as defect
                      and adds the columns 'defect_all' to all Dataframes
    Returns:
        [dict], [dict], [dict]: keys are parameter names, values are
            DataFrames. exp_results contains the data of the calibration step (i.e.
            hardware settings, measured time constants, etc.). cal_results contains
            calibration data (fit function, coefficients, defects, domain). eval_results
            contains measurement results for an evaluation run.
    """
    calib_storage_path = os.path.join(storage_path, calib_name)
    eval_storage_path = os.path.join(storage_path, eval_name)
    with pd.HDFStore(os.path.join(calib_storage_path, 'results.h5')) as store:
        if len(store) == 0 and calib_name != '':
            print "\n", ("WARNING: calibration store for does not contain anything!"
                   " (is the path correct?)  {}".format(calib_storage_path)), "\n"
        exp_results = {key[1:] : store[key] for key in store.keys() if '_calib' not in key}
        cal_results = {key[1:] : store[key] for key in store.keys() if '_calib' in key}
    with pd.HDFStore(os.path.join(eval_storage_path, 'results.h5')) as store:
        if len(store) == 0 and eval_name != '':
            print "\n", ("WARNING: evaluation store does not contain anything!"
                  " (is the path correct?)  {}".format(eval_storage_path)), "\n"
        eval_results = {key[1:] : store[key] for key in store.keys()}
    if patch:
        # add column 'defect_all' to all DataFrames
        patch_defects(cal_results, exp_results)
        patch_defects(cal_results, cal_results)
        patch_defects(cal_results, eval_results)
    return exp_results, cal_results, eval_results

def get_defects(dfs):
    """
    reads out the column 'defect' for all DataFrames in dfs and returns a
    Series, marking defect neurons with True. If no cailbration Dataframe is
    loaded, returns False (no neuron is defect because it has not benn measured).

    Args:
        dfs [list]: list of DataFrames
    Returns:
        [Series, bool]: defect neurons
    """
    if len(dfs) == 0:
        return False
    df_def = []
    for df in dfs.itervalues():
        df_def.append(df['defect'])
    return pd.concat(df_def, axis=1).sum(axis=1) > 0

def patch_defects_redman(results, backend_path, wafer, hicann):
    """
    patch defects using the redman xml file

    Args:
        results [dict]: contains DataFrames to be patched
        backend_path [str]: path to xml file
        wafer [int]: wafer enum
        hicann [int]: hicann enum
    """
    from pyhalbe import Coordinate as C
    import pyredman as redman
    import pycake.helpers.redman
    # first check if file exists, bc. redman does not throw an error
    files = [f for f in os.listdir(backend_path)
             if os.path.isfile(os.path.join(backend_path, f))]
    found = False
    wafer_str = str(wafer)
    hicann_str = str(hicann)
    for f in files:
        if wafer_str in f and hicann_str in f:
            if (f[f.index(wafer_str)-1] not in "123456789"
                    and f[f.index(wafer_str)+len(wafer_str)] not in "0123456789"
                    and f[f.index(hicann_str)-1] not in "123456789"
                    and f[f.index(hicann_str)+len(hicann_str)] not in "0123456789"):
                found = True
    if not found:
        print "\n", ("WARNING: No redman xml found!"
               " (is the path correct?)  {}".format(backend_path)), "\n"
    wafer = C.Wafer(C.Enum(wafer))
    hicann = C.HICANNOnWafer(C.Enum(hicann))
    redman = pycake.helpers.redman.Redman(
        backend_path, C.HICANNGlobal(hicann,wafer))
    neuron_enums = results.values()[0].index.get_level_values('neuron').unique().values
    defects = {ii: not redman.hicann_with_backend.neurons().has(C.NeuronOnHICANN(C.Enum(ii)))
               for ii in neuron_enums}
    df_defect = pd.Series(defects).to_frame()
    df_defect.columns = ['defect_all']
    df_defect.index.names = ['neuron']
    for key, df in results.iteritems():
        index_names = df.index.names
        df = pd.merge(df.reset_index(), df_defect.reset_index(),
                               on=['neuron'], how='inner')
        df.set_index(index_names, inplace=True)
        results[key] = df
    return results

def patch_defects(dfs0, dfs1):
    """
    read in defects from dfs0 and patch them in dfs1

    Args:
        dfs0 [list of DataFrame]: contains info about defect neurons
        dfs1 [list of DataFrame]: gets info about defect neurons
    """
    defects = get_defects(dfs0)
    if isinstance(defects, bool):
        for df in dfs1.itervalues():
            df['defect_all'] = defects
    else:
        defects_val = defects.sortlevel('neuron').values
        for df in dfs1.itervalues():
            if 'step' in df.index.names:
                step_len = len(df.index.get_level_values('step').unique())
                defects = np.repeat(defects_val, step_len)
                df.sortlevel('neuron', inplace=True)
            df['defect_all'] = defects

def get_rep_slice(df, rep, columns):
    """
    reduces Dataframe df to the steps of repetition rep
    TODO: repetitions is inferred from number of different DAC
    values...better if it would be saved during calib
    """
    step_level = df.index.names.index('step')
    df = df.swaplevel(0, step_level).sortlevel('step')
    step_len = min(len(df[col].unique()) for col in columns)
    df = df.loc[range(step_len*rep, step_len*(rep+1)), :]
    df = df.swaplevel(0,step_level)
    return df

def add_FG_Block(df):
    """
    adds FG Block column in df as
      neuron_enum % num_shared_blocks
    + num_neurons_total/num_shared_blocks*shared_block_enum
    """
    num_nrn =  df.index.get_level_values('neuron').unique().shape[0]
    num_shared = df.index.get_level_values('shared_block').unique().shape[0]
    if 'step' in df.index.names:
        num_steps = df.index.get_level_values('step').unique().shape[0]
    else:
        num_steps = 1
    dfs = []
    for block, group in df.groupby(level='shared_block'):
        group.sortlevel('neuron', inplace=True)
        group['FGBlock'] = (np.tile(range(group.shape[0]/num_steps), num_steps)
                            + num_nrn/num_shared*block)
        dfs.append(group.copy())
    return pd.concat(dfs)

def patch_tau_syn(df):
    """
    calculates tau_syn and tau_mem and patches them into the DataFrame
    """
    df['tau_syn'] = df[['tau_1', 'tau_2']].min(axis=1)
    df['tau_mem'] = df[['tau_1', 'tau_2']].max(axis=1)
    return df

def patch_tau_syn_all(df):
    keys = df.keys()
    keys = [k for k in keys if k in ['V_syntci', 'V_syntcx']]
    for key in keys:
        df_to_patch = df[key]
        patch_tau_syn(df_to_patch)

def get_fit_label(df, xunit, yunit):
    """
    returns units of x- and y-axis depending on number of coefficients
    (the coeff is on the x-axis, thus c_i has unit yunit/xunit**i)

    Args:
        df [DataFrame]: contains coefficients
        xunit [str]: x-unit
        yunit [str]: y-unit
    Returns:
        1st argument: [str] the function
        2nd argument: [str] list of units
    """
    trafo_type = [val for val in df['trafo'] if val is not None][0]
    degree = [val for val in df['degree'] if val is not None][0]
    degree = int(degree)
    label = r"c_0"
    units = []
    for ii in range(1, degree+1):
        if ii == 1:
            label += " + c_{}x".format(ii)
            units.append(xunit)
        else:
            label += " + c_{}x^{}".format(ii, ii)
            units.append("{}^{}".format(xunit, ii))
    if 'OneOverPolynomial' == trafo_type:
        units = [r"$\frac{1}{\mathrm{" + yunit + "}" + uu + "}$" for uu in units]
        units.insert(0, r"$\frac{1}{\mathrm{" + yunit + "}}$")
        return r"$\frac{1}{" + label + "}$", units
    else:
        units = [r"$\frac{\mathrm{" + yunit + "}}{" + uu + "}$" for uu in units]
        units.insert(0, r"$\mathrm{" + yunit + "}$")
        return r"${}$".format(label), units

def pad_xaxis(ax):
    """
    pad the xaxis of ax
    axes.margins does not work with pandas sometimes, so it has to be set
    manually
    """
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    xlim = [xlim[0]-0.01*x_range, xlim[1]+0.01*x_range]
    ax.set_xlim(xlim)

def fit_func(df, x_range, cut=False, all_neurons=False, neurons=None):
    """
    calculates fit_function for given DataFrame at points in x_range

    Args:
        df [DataFrame]: contains coefficients, trafo, etc.
        x_range [list, np.ndarray]: contains x values (usually in seconds or
                                    volts)
        cut [bool]: If data should be cut to domain
        all_neurons [bool]: if the fit functions for all neurons in df should
                            be calculated. If not, takes some selected neurons
        neurons [list]: if not None, plot only these neurons

    Returns:
        [DataFrame]: same as df + new axis 'x' and 'y' for x and y values
    """
    trafo_type = [val for val in df['trafo'] if val is not None][0]
    df_fit = df[df['defect']==False]
    coeffs = df_fit.filter(regex='coeff')
    if not all_neurons:
        if neurons is None:
            if trafo_type != 'Lookup':
                # get min and max coeffs and average, to have extreme and normal fits
                neurons = np.concatenate([coeffs.idxmin().values,
                                         coeffs.idxmax().values,
                                         np.abs(coeffs.mean() - coeffs).idxmin().values[:1]])
            else:
                # just draw some neurons bc. the upper selection method
                # would give too many neurons
                neurons = df.index.get_level_values('neuron').unique().values
                nrn_min, nrn_max = np.min(neurons), np.max(neurons)
                lin_neurons = np.linspace(nrn_min, nrn_max, 6, dtype=int)
                neurons = list(neurons[lin_neurons])
        df_fit = df_fit.loc[neurons,:]
    dfs = []
    for x in x_range:
        df_fit['x'] = x
        dfs.append(df_fit.copy())
    df_fit = pd.concat(dfs)
    if cut:
        df_fit['cut_min'] = df_fit['domain_min'] > df_fit['x']
        df_fit['cut_max'] = df_fit['domain_max'] < df_fit['x']
        df_fit['cut'] = df_fit[['cut_min', 'cut_max']].sum(1) > 0
        df_fit = df_fit[~df_fit['cut']]

    if trafo_type in ['Constant', 'Polynomial', 'OneOverPolynomial']:
        df_fit['y'] = 0.
        for ii,coeff in enumerate(coeffs):
            df_fit['y'] += df_fit[coeff]*df_fit['x']**ii
        if trafo_type == 'OneOverPolynomial':
            df_fit['y'] = 1./df_fit['y']

    elif trafo_type == 'Lookup':
        results = []
        for x, df_neuron in df_fit.groupby(['x']):
            result = df_neuron.filter(regex='coeff')
            result = np.abs(result.subtract(df_neuron['x'], axis='index')).idxmin(1)
            result = result.apply(lambda x: int(x[5:]))
            df_neuron['y'] = result
            results.append(df_neuron)
        df_fit = pd.concat(results)
    else:
        raise LookupError('No fit function definded for trafo type {}'.format(trafo_type))
    return df_fit

def load_trace(storage_path, cal_eval_name, parameter, nrn):
    """
    load the traces which are stored in a hdf5 store at storage_path/eval_name
    # TODO: There is no way to savely determine which files contain which
    # traces (different parameter traces are in different folders but they are
    # named 0, 1 etc. Thus it is assumed that the V_reset measurement always
    # comes before the V_t measurement and traces are read in this order.

    Args:
        storage_path [str]: path to the hdf5 file of the traces
        cal_eval_name [str]: name of calibration/evaluation folder
        parameter [str]: name of parameter of which the trace should be loaded
        nrn [int]: return trace of this neuron
    Returns:
        [list]: DataFrames that contain traces
    """
    # find folders that contain traces
    paths = []
    data_path = os.path.join(storage_path, cal_eval_name)
    if os.path.isdir(data_path):
        for file_store in os.listdir(data_path):
            file_path = os.path.join(data_path, file_store)
            if os.path.isdir(file_path):
                for files in os.listdir(file_path):
                    if '.hdf5' in files:
                        paths.append(file_path)
                        break
    if nrn < 10:
        nrn = "00{}".format(nrn)
    elif nrn < 100:
        nrn = "0{}".format(nrn)
    else:
        nrn = "{}".format(nrn)
    if parameter == 'V_reset':
        path = paths[0]
    elif parameter == 'V_t':
        path = paths[1]
    else:
        raise LookupError("No traces exist for this parameter")
    dfs = []
    for file_store in os.listdir(path):
        if ".hdf5" in file_store:
            store_path = os.path.join(path, file_store)
            with pd.HDFStore(store_path) as store:
                dfs.append(store['trace_{}'.format(nrn)])
    return dfs

def plot_trace(dfs, nrn, **kwargs):
    """
    plot voltage traces (data from load_trace)
    """
    ax = kwargs.get('ax', None)
    if ax == None:
        fig, ax = plt.subplots()
        kwargs.update({'ax' : ax})
    for df in dfs:
        df.plot(**kwargs)

    if ax.legend() is not None:
        ax.legend().set_visible(False)
    ax.set_title("Trace of Neuron {}".format(nrn))
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$V_{membrane}$ [V]")
    if locals().get('fig', None) is None:
        return ax
    else:
        return fig, ax

def plot_defects(df, all_defects=True, **kwargs):
    """
    plot defect and working neurons shared block vs neuron

    Args:
        all_defects [bool]: If defects of all parameters is taken. Otherwise,
        take only defect neurons of parameter of the DataFrame
    """
    std_kwargs = {'linestyle' : '', 'marker' : 'x'}
    for key in std_kwargs.keys():
        if key not in kwargs.keys():
            kwargs.update({key : std_kwargs[key]})

    ax = kwargs.get('ax', None)
    if ax == None:
        fig, ax = plt.subplots()
        kwargs.update({'ax' : ax})
    defect_column = 'defect'
    if all_defects:
        defect_column += '_all'
    defects = df[defect_column]
    defects = defects.reset_index(['neuron', 'shared_block'])
    defects['y'] = defects['shared_block'] + defects[defect_column]/2.
    defects[defects[defect_column]==False].plot('neuron', 'y', color='g',
                                                label='working', **kwargs)
    if defects[defects[defect_column]==True].shape[0] != 0:
        defects[defects[defect_column]==True].plot('neuron', 'y', color='r',
                                                   label='defect', **kwargs)
    # This is a workaround bc. one marker in the legend is missing (pandas plot bug)
    lines = ax.get_lines()
    ax.legend(lines, ['Working', 'Defect'])

    num_fg_blocks = defects['shared_block'].unique().shape[0]
    ax.set_yticks([rr + 0.25 for rr in range(num_fg_blocks)])
    ax.set_yticklabels(range(num_fg_blocks))
    plt_min_max = [defects['neuron'].min(), defects['neuron'].max()]
    ax.set_xlim(plt_min_max)
    ax.set_xlabel("neuron")
    ax.set_ylabel("FGBlock")
    if locals().get('fig', None) is None:
        return ax
    else:
        return fig, ax

def plot(df, columns, level='neuron', with_defects=True,
         all_defects=True, rep=None, refs=None, set_tb=False, per_fg_block=False, **kwargs):
    """
    plot a line or markers. Takes the columns 'columns' to plot df[columns]

    Args:
        df [DataFrame]: contains the data
        columns [list]: x- and y-axis of plot
        level [string]: which level to sort by: 'step' makes a plot for every
                        step (on the same figure), 'neuron' for every neuron etc.
        with_defects [bool]: if True, include defect neurons, otherwise exclude
        all_defects [bool]: include all defects from different parameters,
                            otherwise include only defects from current
                            parameter (defined by the DataFrame
        rep [int]: plots only the steps with repetition 'rep'
        refs [list]: plots reference lines (for eval)
        set_tb [bool]: for plots where tob/bottom neurons are plotted separately
        per_fg_block [bool]: if True, data is plotted vs FG Blocks
        kwargs: pandas.plot kwargs

    Returns:
        axis or figure, axis if axis was None
    """
    first_bottom = 256 #first bottom neuron on HICANN chip
    std_kwargs = {'alpha' : 0.05, 'linestyle' : '-', 'marker' : '.', 'legend' : None}
    for key in std_kwargs.keys():
        if key not in kwargs.keys():
            kwargs.update({key : std_kwargs[key]})
    if set_tb:
        kwargs.update({'color' : 'b'})

    ax = kwargs.get('ax', None)
    if ax == None:
        fig, ax = plt.subplots()
        kwargs.update({'ax' : ax})
    df_plot = df.copy(deep=True)
    # insert a column for the index, without removing the index
    for col in columns:
        if col in df_plot.index.names:
            df_plot.reset_index(col, inplace=True)
            df_plot.set_index(col, drop=False, append=True, inplace=True)

        if all_defects is False:
            defect_str = 'defect'
        else:
            defect_str = 'defect_all'
        if with_defects is False:
            df_plot = df_plot[columns + [defect_str]]
            df_plot  = df_plot[df_plot[defect_str]==False]
            neurons = df_plot.index.get_level_values('neuron')
        else:
            df_plot = df_plot.loc[:, columns + [defect_str]]
    if rep is not None:
        df_plot = get_rep_slice(df_plot, rep, columns)
    df_plot.sortlevel(level)
    if per_fg_block:
        df_plot = add_FG_Block(df_plot)
        columns[0] = 'FGBlock'

    if level is None:
        df_plot.plot(x=columns[0], y=columns[1], **kwargs)
        if with_defects:
            col, al = kwargs.get('color', None), kwargs.get('alpha', None)
            kwargs['color'] = 'red'
            kwargs['alpha'] = 1.
            if df_plot[df_plot[defect_str]].shape[0] != 0:
                df_plot[df_plot[defect_str]].plot(x=columns[0],
                                                  y=columns[1], **kwargs)
            kwargs['color'] = col
            kwargs['alpha'] = al
    else:
        for name, group in df_plot.groupby(level=level):
            if name >= first_bottom and set_tb:
                kwargs.update({'color' : 'g'})
            group.plot(x=columns[0], y=columns[1], **kwargs)
            if with_defects:
                col, al = kwargs.get('color', None), kwargs.get('alpha', None)
                kwargs['color'] = 'red'
                kwargs['alpha'] = 1.
                if group[group[defect_str]].shape[0] != 0:
                    group[group[defect_str]].plot(x=columns[0],
                                                  y=columns[1], **kwargs)
                kwargs['color'] = col
                kwargs['alpha'] = al

    if refs is not None:
        for ref in refs:
            ax.axhline(ref, ls='--', color='k')
    if set_tb:
        # Due to high alpha, set legend values by hand
        line_top = mlines.Line2D([], [], color='blue', marker='.',
                                 label='top')
        line_bottom = mlines.Line2D([], [], color='green', marker='.',
                                 label='bottom')
        plt.legend(handles=[line_top, line_bottom])
        ax.legend([line_top, line_bottom], ['top', 'bottom'])
    ax.grid(True)
    if per_fg_block:
        num_nrn =  df_plot.index.get_level_values('neuron').unique().shape[0]
        num_fg_blocks = df_plot.index.get_level_values('shared_block').unique().shape[0]
        fg_pos = [rr*num_nrn/num_fg_blocks for rr in range(num_fg_blocks)]
        ax.set_xticks(fg_pos)
        ax.set_xticklabels(range(num_fg_blocks))
    if locals().get('fig', None) is None:
        return ax
    else:
        return fig, ax

def hist(df, column, x_label=None, with_defects=True, all_defects=True,
         ref=None, plot_mean=True, **kwargs):
    """
    plot a histogram of the data df[column].

    Args:
        df [DataFrame]: contains data
        column [string]: column to plot
        x_label [string]: label of the x-axis. Also used to extract unit
        with_defects [bool]: if True, include defect neurons, otherwise exclude
        all_defects [bool]: include all defects from different parameters,
                            otherwise include only defects from current
                            parameter (defined by the DataFrame
        refs [list]: plots reference lines (for eval)
        plot_mean [bool]: plots a vertical line at the mean of the histogram
        kwargs: pandas.plot kwargs

    Returns:
        axis or figure, axis if axis was None
    """
    std_kwargs = {'bins' : 100}
    for key in std_kwargs.keys():
        if key not in kwargs.keys():
            kwargs.update({key : std_kwargs[key]})
    ax = kwargs.get('ax', None)
    if ax == None:
        fig, ax = plt.subplots()
        kwargs.update({'ax' : ax})

    if with_defects is False:
        if all_defects is False:
            num_not_defect = len(df[~df['defect']])
            df_plot = df[df['defect']==False][column]
        else:
            num_not_defect = len(df[~df['defect_all']])
            df_plot = df[df['defect_all']==False][column]
    else:
        df_plot = df[column]
        num_not_defect = len(df.index.get_level_values('neuron').unique())
    mean = df_plot.mean()
    std = df_plot.std()
    if 'label' not in kwargs.keys() and x_label is not None:
        s0,s1 = x_label.find('['), x_label.find(']')+1
        # ignore if np.log10 returns inf
        oldsettings = np.seterr(divide='ignore')
        if np.log10(np.abs(mean)) < -4:
            mean_label = mean*1e6
            std_label = std*1e6
            ref_label = ref*1e6 if ref is not None else None
            x_unit = r"$\mu$" + x_label[s0+1:s1-1]
        else:
            mean_label = mean
            std_label = std
            ref_label = ref
            x_unit = x_label[s0+1:s1-1]
        label = "{:.4g} $\pm$ {:.2g} {}, {}".format(mean_label, std_label, x_unit, num_not_defect)
        np.seterr(**oldsettings)
        if ref is not None:
            label += " ({} {})".format(ref_label, x_unit)
        kwargs.update({'label': label})

    df_plot.plot(kind='hist', **kwargs)
    if plot_mean:
        color = ax.patches[-1].get_facecolor()
        ax.axvline(mean, ls='--', color=color)
    if ref is not None:
        ax.axvline(ref, ls='--', color='k')
    ax.set_xlabel(x_label)

    if locals().get('fig', None) is None:
        return ax
    else:
        return fig, ax

def hists(df, column, level, x_label=None, with_defects=True, all_defects=True,
          refs=None, plot_mean=True, num_steps=None, **kwargs):
    """
    plot multiple histograms

    Args:
        df [DataFrame]: contains data
        column [string]: column to plot
        x_label [string]: label of the x-axis. Also used to extract unit
        with_defects [bool]: if True, include defect neurons, otherwise exclude
        all_defects [bool]: include all defects from different parameters,
                            otherwise include only defects from current
                            parameter (defined by the DataFrame)
        refs [list]: plots reference lines (for eval)
        plot_mean [bool]: plots a vertical line at the mean of the histogram
        num_steps [int]: max number of steps to plot
        kwargs: pandas.plot kwargs

    Returns:
        axis or figure, axis if axis was None
    """
    if num_steps is not None:
        df_plot = df.swaplevel(0, 'step').sortlevel('step').loc[:num_steps]
    else:
        df_plot = df
    for ii, (_, group) in enumerate(df_plot.sortlevel(level).groupby(level=level)):
        if refs is None:
            hist(group, column, x_label=x_label, with_defects=with_defects,
                 ref=None, plot_mean=plot_mean, **kwargs)
        else:
            hist(group, column, x_label=x_label, with_defects=with_defects,
                 ref=refs[ii], plot_mean=plot_mean, **kwargs)

def plot_domains(df, **kwargs):
    """
    plot domains

    Args:
        df [DataFrame]: contains domain_min and domain_max
    Returns:
        axis or figure, axis if axis was None
    """
    ax = kwargs.get('ax', None)
    if ax == None:
        fig, ax = plt.subplots()
        kwargs.update({'ax' : ax})
    df_plot = df[['domain_min', 'domain_max']]
    df_plot.loc[:,'diff'] = ((df_plot.loc[:,'domain_max']
                            - df_plot.loc[:,'domain_min']) / 2.)
    df_plot = df_plot.sort_values('diff', ascending=False)
    df_plot.loc[:, 'mean'] = df_plot.loc[:,['domain_min', 'domain_max']].mean(axis=1)
    df_plot.reset_index(inplace=True)
    df_plot.dropna().plot(y='mean', yerr='diff', linestyle='', **kwargs)

    ax.legend().set_visible(False)
    mean = df_plot['mean'].mean()
    ax.set_xlabel('neuron')
    ax.set_ylabel('time domain [s]')
    nrn_num = df.index.get_level_values('neuron').unique().shape[0]
    ax.set_xlim([0, nrn_num])

    if locals().get('fig', None) is None:
        return ax
    else:
        return fig, ax

def plot_spikes(df, **kwargs):
    """
    plot mean spikes (for Spikes eval)

    Returns:
        axis or figure, axis if axis was None
    """
    ax = kwargs.get('ax', None)
    if ax == None:
        fig, ax = plt.subplots()
        kwargs.update({'ax' : ax})
    for tb in ['top', 'bottom']:
        nrns = range(256) if tb is 'top' else range(256,512)
        color = 'b' if tb is 'top' else 'g'
        cols = ['V_t_config', 'spikes_n_spikes']
        mean = df.sortlevel('neuron').loc[nrns, cols].mean(0, level='step')
        std = df.sortlevel('neuron').loc[nrns, cols].std(0, level='step')
        df_plot = pd.concat([mean, std], 1)
        df_plot.columns = ['V_t', 'y', 'xerr', 'yerr']
        df_plot.plot('V_t', 'y', yerr='yerr', label=tb, marker='o', color=color, **kwargs)

    if locals().get('fig', None) is None:
        return ax
    else:
        return fig, ax

def plot_spiking_neurons(df, with_defects=True, ax=None, **kwargs):
    """
    plot mean spikes over V_t (for Spikes eval)

    Returns:
        axis or figure, axis if axis was None
    """
    if ax == None:
        fig, ax = plt.subplots()
    df_plot = df[['V_t_config', 'spikes_n_spikes', 'defect_all']]
    if with_defects is False:
        df_plot = df_plot[df_plot['defect_all']==False]
    num_nrns = df_plot.index.get_level_values('neuron').unique().shape[0]
    num_spikes = []
    for _, group in df_plot.groupby(level='step'):
        num_spikes.append([group['V_t_config'].iloc[0],
                         (group['spikes_n_spikes'] > 1).sum()])
    ax.plot(*zip(*num_spikes), **kwargs)
    ax.axhline(num_nrns, linestyle='--', color='k')

    if locals().get('fig', None) is None:
        return ax
    else:
        return fig, ax

def plot_coefficients(df, key, with_defects, axes=None, **kwargs):
    config = plot_config[key[:-6]]
    coeffs = [col for col in df.columns if 'coeff' in col]
    title, units = get_fit_label(df, xunit=config['xunit'],
                                 yunit=config.get('yunit', 'DAC'))
    num_plots = len(coeffs)
    if axes == None:
        fig, axes = plt.subplots(2, num_plots, figsize=(16., 4.8))
        fig.suptitle("Calibration coefficients", y=1.)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    for ii, coeff in enumerate(coeffs):
        hist(df, coeff, x_label=r"$c_{}$ [{}]".format(ii, units[ii]),
             with_defects=with_defects, ax=axes[ii], bins=150, **kwargs)
        axes[ii].legend(loc='best').get_frame().set_alpha(0.5)

        if config.get('per_fg_block', False):
            label_nrn = 'shared_FG_block'
        else:
            label_nrn = 'neuron_number'
        plot(df, ['neuron', coeff], ax=axes[ii+num_plots],
             level=None, with_defects=with_defects,
             per_fg_block=config.get('per_fg_block', False),
             marker='x', alpha=1., linestyle='', **kwargs)
        axes[ii+num_plots].set_ylabel(r"$c_{}$ [{}]".format(ii, units[ii]))
    axes[0].legend(loc='best', title=title).get_frame().set_alpha(0.5)
    if config.get('DACrange', False):
        axes[0].set_xlim(config['DACrange'])
        axes[num_plots].set_ylim(config['DACrange'])
    return fig, axes

def main():
    import shutil
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("storage_path", help="path of calibration and evaluation directory")
    parser.add_argument("--calib", help="name of calibration directory", default='')
    parser.add_argument("--evaluation", help="name of evaluation directory", default='')
    parser.add_argument("--outdir", help="path of output directory for plots",
                        default="figures_pandas")
    parser.add_argument("--neuron_enum", nargs='+', type=int, help="neurons for"
                        "which the trace of V_reset and V_t should be plotted",
                        default=None)
    parser.add_argument("--wafer", help="Wafer enum", type=int,
                        default=None)
    parser.add_argument("--hicann", help="HICANN enum", type=int,
                        default=None)
    parser.add_argument("--backend_path", help="path to the folder containing the redman xml",
                        default=".")

    args = parser.parse_args()

    # insert neuron enums (to plot traces of) in config dict
    if args.neuron_enum:
        for config in plot_config.itervalues():
            config['nrns'] = args.neuron_enum
        neuron_enums = args.neuron_enum

    storage_path = args.storage_path
    calib_name = args.calib
    eval_name = args.evaluation
    fig_dir_name = args.outdir
    # parameters only needed if no calib data is given (to get defects for
    # eval)
    wafer = args.wafer
    hicann = args.hicann
    backend_path = args.backend_path
    if wafer is not None and hicann is not None:
        patch = False
    else:
        patch = True

    # create fig directory if not already there
    fig_dir_path = make_fig_dir(storage_path, fig_dir_name)
    cakebin = os.path.split(os.path.abspath(__file__))[0]
    shutil.copy(os.path.join(cakebin, "overview.html"), fig_dir_path)
    # trace plots have individual neuron values in file name, add
    with open(os.path.join(fig_dir_path, "overview.html"), 'rw') as overview:
        data = overview.readlines()
        remove = True
        for nrn in neuron_enums:
            if nrn == 0:
                remove = False
            else:
                count = 0
                for ii, line in enumerate(data):
                    mod = count % 2
                    if "nrn_0" in line and mod == 0:
                        count += 1
                        newline = line.replace('0', str(nrn))
                        newline1 = data[ii+1].replace('0', str(nrn))
                        data.insert(ii+2, newline)
                        data.insert(ii+3, newline1)
                    elif "nrn_0" in line and mod != 0:
                        count += 1
        if remove == True:
            for ii, line in enumerate(data):
                if "nrn_0" in line:
                    data[ii] = "\n"
    with open(os.path.join(fig_dir_path, "overview.html"), 'w') as overview:
        overview.writelines(data)

    # load data
    exp_results, cal_results, eval_results = load_data(storage_path, calib_name,
                                                       eval_name, patch=patch)
    # if patch is False, patch from redman
    if not patch:
        for results in [exp_results, cal_results, eval_results]:
            if results:
                results = patch_defects_redman(results, backend_path, wafer, hicann)

    # patch tau_syn
    patch_tau_syn_all(exp_results)
    fig, ax = plt.subplots(figsize=(16,5.))
    # plot defects, they are patched in every DataFrame so we can take any
    if len(exp_results) > 0:
        plot_defects(exp_results.values()[0], ax=ax)
    else:
        plot_defects(eval_results.values()[0], ax=ax)
    fig_filename = 'defect_neurons_vs_neuron_number.png'
    fig_path = os.path.join(fig_dir_path, fig_filename)
    print "Saving {}.".format(fig_filename)
    plt.savefig(fig_path)
    ax.cla()


    fig, ax = plt.subplots()

    # plot readout_shift
    if 'readout_shift_calib' in cal_results.keys():
        df = cal_results['readout_shift_calib']
        config = plot_config['readout_shift']
        result_key = config['result_key']

        hist(df, result_key, x_label=config['result_label'], plot_mean=True, ax=ax)
        fig_filename = 'analog_readout_offset.png'
        fig_path = os.path.join(fig_dir_path, fig_filename)
        print "Saving {}.".format(fig_filename)
        plt.savefig(fig_path)
        ax.cla()

        plot(df, ['neuron', result_key], ax=ax,
             level=None, alpha=1., linestyle='', marker='x')
        ax.set_xlim([0,512])
        ax.set_ylabel(config['result_label'])
        fig_filename = 'analog_readout_offset_vs_nrn.png'
        fig_path = os.path.join(fig_dir_path, fig_filename)
        print "Saving {}.".format(fig_filename)
        plt.savefig(fig_path)
        ax.cla()

    # plot domains
    keys = cal_results.keys()
    keys = [k for k in keys if k in ['V_syntci_calib', 'V_syntcx_calib']]
    for key in keys:
        df = cal_results[key]
        plot_domains(df, ax=ax)
        fig_filename = '{}_domain.png'.format(key[:-6])
        fig_path = os.path.join(fig_dir_path, fig_filename)
        print "Saving {}.".format(fig_filename)
        plt.savefig(fig_path)
        ax.cla()

    # plot effective resting potential
    columns = ['I_gl', 'mean']
    for with_defects in [True, False]:
        label_wd = 'with_defects' if with_defects else 'without_defects'
        keys = eval_results.keys()
        keys = [k for k in keys if k in ['V_convoff_test_calibrated',
                                         'V_convoff_test_uncalibrated']]
        for key in keys:
            label_cal = 'uncalibrated' if 'uncalibrated' in key else 'calibrated'
            df = eval_results[key]
            plot(df, columns, ax=ax, level=None,
                 with_defects=with_defects, marker='x', linestyle='', alpha=0.1, color='k')
            ax.set_xlim([-10,1023])
            ax.set_xlabel(r"$I_{gl}$ [DAC]")
            ax.set_ylabel(r"effective resting potential [V]")
            for I_gl, tmpdata in df.groupby(columns[0]):
                mean = tmpdata[columns[1]].mean()
                std = tmpdata[columns[1]].std()
                ax.text(I_gl, 1.10,
                         r"${:.0f} \pm {:.0f}$ mV".format(mean * 1000, std * 1000),
                         rotation=35)
            pad_xaxis(ax)
            fig_filename = 'V_convoff_{}_test_{}.png'.format(label_wd, label_cal)
            fig_path = os.path.join(fig_dir_path, fig_filename)
            print "Saving {}.".format(fig_filename)
            plt.savefig(fig_path)
            ax.cla()

    # plot spikes
    for with_defects in [True, False]:
        label_wd = 'with_defects' if with_defects else 'without_defects'
        if 'Spikes' in eval_results.keys():
            df = eval_results['Spikes']
            print "Plotting {}, {}".format('Spikes', label_wd)

            plot_spikes(df, ax=ax)
            pad_xaxis(ax)
            fig_filename = 'n_spikes_{}_result_calibrated.png'.format(label_wd)
            fig_path = os.path.join(fig_dir_path, fig_filename)
            print "Saving {}.".format(fig_filename)
            plt.savefig(fig_path)
            ax.cla()

            plot_spiking_neurons(df, ax=ax, with_defects=with_defects)
            pad_xaxis(ax)
            fig_filename = 'n_spiking_neurons_{}_calibrated.png'.format(label_wd)
            fig_path = os.path.join(fig_dir_path, fig_filename)
            print "Saving {}.".format(fig_filename)
            plt.savefig(fig_path)
            ax.cla()

    # plot tau_syn, tau_mem vs V_syntc
    for with_defects in [True, False]:
        label_wd = 'with_defects' if with_defects else 'without_defects'
        keys = exp_results.keys()
        keys = [k for k in keys if k in ['V_syntci', 'V_syntcx']]
        for key in keys:
            print "Plotting {}, {}".format(key, label_wd)
            df = exp_results[key]

            config = plot_config[key]
            for tau in ['tau_syn', 'tau_mem']:
                plot(df, [key, tau], ax=ax, with_defects=with_defects, set_tb=True)
                if tau is 'tau_syn':
                    fig_filename = '{}_{}_result_uncalibrated.png'.format(key, label_wd)
                else:
                    fig_filename = '{}_{}_result_tau_mem_uncalibrated.png'.format(key, label_wd)
                pad_xaxis(ax)
                ax.set_xlabel(config['key_label_cal'])
                if tau is 'tau_syn':
                    ax.set_ylabel(r"$\tau_{syn}$ [s]")
                else:
                    ax.set_ylabel(r"$\tau_{mem}$ [s]")
                fig_path = os.path.join(fig_dir_path, fig_filename)
                print "Saving {}".format(fig_filename)
                plt.savefig(fig_path)
                ax.cla()

        keys = eval_results.keys()
        keys = [k for k in keys if k in ['V_syntci', 'V_syntcx']]
        for key in keys:
            print "Plotting {}, {}".format(key, label_wd)
            df = eval_results[key]
            patch_tau_syn(df)

            config = plot_config[key]
            refs = df[key + '_config'].unique()
            hists(df, config['result_key'], level='step', ax=ax, bins=50, refs=refs,
                  x_label=config['result_label'], with_defects=with_defects)
            ax.legend(loc='best').get_frame().set_alpha(0.5)
            fig_filename = '{}_{}_{}.png'.format(key, label_wd, 'calibrated')
            fig_path = os.path.join(fig_dir_path, fig_filename)
            print "Saving {}".format(fig_filename)
            plt.savefig(fig_path)
            ax.cla()


    # plot the remaining plots
    for cal_or_eval in ['cal', 'eval']:
        keys = exp_results.keys() if cal_or_eval in 'cal' else eval_results.keys()
        keys = [k for k in keys if k in ['V_reset', 'V_t', 'E_synx',
                                         'E_syni', 'E_l', 'I_pl', 'I_gl_PSP']]
        label_cal = 'uncalibrated' if cal_or_eval == 'cal' else 'calibrated'
        for with_defects in [True, False]:
            label_wd = 'with_defects' if with_defects else 'without_defects'

            for key in keys:
                load_key = key if key != 'I_gl_PSP' else 'I_gl'
                df = exp_results[key] if cal_or_eval in 'cal' else eval_results[key]
                print "Plotting {}, {}, {}".format(key, label_cal, label_wd)

                config = plot_config[load_key]
                result_key = config['result_key']
                result_label = config['result_label']
                if cal_or_eval in 'cal':
                    columns = [load_key, result_key]
                else:
                    columns = [load_key+'_config', result_key]
                plot(df, columns, ax=ax, with_defects=with_defects, set_tb=True,
                     rep=config.get('rep', None))
                ax.set_xlabel(config['key_label_'+cal_or_eval])
                ax.set_ylabel(config['result_label'])
                pad_xaxis(ax)
                fig_filename = '{}_{}_result.png'.format(key, label_wd)
                if cal_or_eval == 'eval':
                    fig_filename = fig_filename[:-4]
                    fig_filename += '_{}.png'.format(label_cal)
                fig_path = os.path.join(fig_dir_path, fig_filename)
                print "Saving {}".format(fig_filename)
                plt.savefig(fig_path)
                ax.cla()

                if key == 'I_pl' and cal_or_eval == 'cal':
                    # I_pl is given as DAC values in v4_params, thus there are no refs
                    refs = None
                else:
                    refs = df[load_key+'_config'].unique() if load_key+'_config' in df.columns \
                                                           else None
                hists(df, result_key, level='step', ax=ax, bins=config.get('bins', 50),
                      logx=config.get('logx', False), with_defects=with_defects, refs=refs,
                      num_steps=config.get('num_steps', None), x_label=config['result_label'])
                ax.legend(loc='best').get_frame().set_alpha(0.5)
                fig_filename = '{}_{}_{}.png'.format(key, label_wd, label_cal)
                fig_path = os.path.join(fig_dir_path, fig_filename)
                print "Saving {}".format(fig_filename)
                plt.savefig(fig_path)
                ax.cla()

                if config.get('per_fg_block', False):
                    label_nrn = 'shared_FG_block'
                else:
                    label_nrn = 'neuron_number'
                plot(df, ['neuron', result_key], level='step', ax=ax,
                     refs=refs, with_defects=with_defects,
                     per_fg_block=config.get('per_fg_block', False),
                     **{'marker': 'x', 'alpha': 1., 'linestyle': ''})
                ax.set_ylabel(config['result_label'])
                fig_filename = '{}_{}_{}_vs_{}.png'.format(key, label_wd, label_cal, label_nrn)
                fig_path = os.path.join(fig_dir_path, fig_filename)
                print "Saving {}".format(fig_filename)
                plt.savefig(fig_path)
                ax.cla()

                if key in ['V_reset', 'V_t']:
                    name = calib_name if cal_or_eval == 'cal' else eval_name
                    nrns = config.get('nrns', [0])
                    for nrn in nrns:
                        dfs = load_trace(storage_path, name,
                                         key, nrn=nrn)
                        if dfs != []:
                            plot_trace(dfs, nrn=nrn, ax=ax, xlim=config.get('xlim', [0, 21e-6]))
                            fig_filename = '{}_trace_{}_nrn_{}.png'.format(key, label_cal,
                                                                           nrn)
                            fig_path = os.path.join(fig_dir_path, fig_filename)
                            print "Saving {}".format(fig_filename)
                            plt.savefig(fig_path)
                            ax.cla()

    # plot fits
    keys = exp_results.keys()
    keys = [k for k in keys if k not in ['readout_shift', 'V_convoffx', 'V_convoffi']]

    for key in keys:
        load_key = key if key != 'I_gl_PSP' else 'I_gl'
        config = plot_config[load_key]
        result_key = config['result_key']
        result_label = config['result_label']
        df = cal_results[load_key + '_calib']
        x_range = config.get('xrange', np.linspace(0.2, 1.5, 100))
        df_fit = fit_func(df, x_range=x_range, cut=False, all_neurons=False)
        plot(df_fit, ['x', 'y'], with_defects=False, ax=ax,
             logx=config.get('logx', False), marker='', alpha=1.)

        df = exp_results[key]
        neurons = list(df_fit.index.get_level_values('neuron').unique().values)
        ax.legend(ax.get_lines(), neurons, title='neuron')
        df = df.sortlevel('neuron').loc[neurons,:]
        columns = [result_key, load_key]
        ax.set_prop_cycle(None) # reset the color cycle
        plot(df, columns, ax=ax, with_defects=False,
             rep=config.get('rep', None), alpha=1.)
        ax.set_ylim([-20, 1050])
        ax.set_xlabel(config['result_label'])
        ax.set_ylabel(config['key_label_cal'])
        pad_xaxis(ax)
        fig_filename = '{}_fit.png'.format(load_key)
        fig_path = os.path.join(fig_dir_path, fig_filename)
        print "Saving {}.".format(fig_filename)
        plt.savefig(fig_path)
        ax.cla()

        df = cal_results[load_key + '_calib']
        x_range = config.get('xrange', np.linspace(0.2, 1.5, 100))
        df_fit = fit_func(df, x_range=x_range, cut=False, all_neurons=True)
        plot(df_fit, ['x', 'y'], with_defects=False, ax=ax,
             logx=config.get('logx', False), marker='')
        ax.set_ylim([-20, 1050])
        ax.set_xlabel(config['result_label'])
        ax.set_ylabel(config['key_label_cal'])
        pad_xaxis(ax)
        fig_filename = '{}_fit_all.png'.format(load_key)
        fig_path = os.path.join(fig_dir_path, fig_filename)
        print "Saving {}.".format(fig_filename)
        plt.savefig(fig_path)
        ax.cla()

    # plot coefficients
    for with_defects in [True, False]:
        label_wd = 'with_defects' if with_defects else 'without_defects'
        for key, df in cal_results.iteritems():
            if df[~df['defect_all']]['trafo'].iloc[0] not in 'Lookup':
                fig, axes = plot_coefficients(df, key, with_defects)

                fig_filename = '{}_{}_hist.png'.format(key, label_wd)
                fig_path = os.path.join(fig_dir_path, fig_filename)
                print "Saving {}".format(fig_filename)
                plt.savefig(fig_path)
                for ax in axes:
                    ax.cla()


if __name__ == "__main__":
    main()
