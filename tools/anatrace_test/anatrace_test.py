import optparse as op
import json
import numpy as np
import scipy.signal as scsig
import scipy.stats as scstat
import pylogging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import mkdir
import os.path

parser = op.OptionParser()
parser.add_option("--wnr", default=33, dest="wnr", type="int", help="wafer number")
parser.add_option("--hnr", default=297, dest="hnr", type="int", help="hicann number")
parser.add_option("--anr", default=0, dest="anr", type="int", help="analog out number")
(opts, args) = parser.parse_args()

def isclose(a, b, tol):
    return (a > b*(1.-tol)) and (a < b*(1.+tol))

def is_overtone(base, testfreq, tol):
    x = testfreq / base
    xr = np.round(x)
    if isclose(x, xr, tol) and xr > 1.5 and xr < 21.5:
        return int(xr)
    else:
        return False

def minmax_idx(a, tol, da):
    return (int(a * (1 - tol) / da), int(a * (1 + tol) / da))

def rms(v):
    return np.sqrt(np.mean(np.square(v)))

def d(t):
    return (t[-1] - t[0]) / np.size(t)

def fftit(t, dat, window):
    datlen = np.size(dat)
    if datlen % 2 == 1:
        flen = (datlen - 1) // 2 + 1
    else:
        flen = datlen // 2 + 1
    f_ = 1. / d(t)
    if window == "rect":
        return (np.linspace(0, f_/2., flen), np.fft.rfft(dat) / np.sqrt(datlen))
    elif window == "flattop":
        w = scsig.flattop(datlen)
    elif window == "hamming":
        w = scsig.hamming(datlen)
    else:
        return None
    return (np.linspace(0, f_/2., flen), np.fft.rfft(np.multiply(dat, w)) / np.sqrt(np.sum(w)))

def dB(fftdat):
    return 20*np.log10(np.abs(fftdat))

pylogging.default_config(date_format='absolute')
pylogging.set_loglevel(pylogging.get("anatrace_test"), pylogging.LogLevel.INFO)
logger = pylogging.get("anatrace_test")

t, v = np.load("membrane_w{}_h{}_a{}.npy".format(opts.wnr, opts.hnr, opts.anr))
tlen = 960000
if (np.size(t) < tlen):
    logger.warn("trace too short")
    quit()
else:
    t = t[:tlen]
    v = v[:tlen]
v_mean = np.mean(v)
# remove dc content
v -= v_mean
v_rms = rms(v)
if (v_rms < 5.e-3):
    logger.info("rms of voltage trace {:.1f} mV".format(v_rms*1.e3))
elif (v_rms < 50.e-3):
    logger.warn("rms of voltage trace is large {:.1f} mV".format(v_rms*1.e3))
else:
    logger.error("rms of voltage trace is unreasonable {:.1f} mV -> check if correct stable membrane data".format(v_rms*1.e3))
    quit()

high_v_ratio = (np.size(np.where(v > 4.*v_rms)) + np.size(np.where(v < 4.*v_rms))) / (2.*scstat.norm.sf(4.)*np.size(v))
if high_v_ratio < 2.:
    logger.info("voltage distribution is sufficiently flat at 4*v_rms range")
elif high_v_ratio < 5.:
    logger.warn("high number of samples outside of 4*v_rms range -> switching peaks?")
else:
    logger.error("extreme high number of samples outside of 4*v_rms range -> switching peaks!")

# rectangular window for power spectrum
f_p, fft_p = fftit(t, v, "rect")
fft_p_global_baseline = np.median(np.abs(fft_p))
# flat-top window for accurate amplitude of peaks
f, fft = fftit(t, v, "flattop")
fft_abs = np.abs(fft)
fft_global_baseline = np.median(fft_abs)
fft_global_max = np.max(fft_abs)

rel_prominence_thres = 0.05 # magic number
prominence_thres = rel_prominence_thres * (fft_global_max - fft_global_baseline)
logger.info("setting prominence threshold of peak detection to {:.2f} ({:.1f}% of global maximum)".format(prominence_thres, rel_prominence_thres*100.))
rel_wlen = 1./5000. # magic number
wlen = int(np.size(f) * rel_wlen)
if (wlen < 100):
    wlen = 100
    rel_wlen = float(wlen) / np.size(f)
logger.info("setting window size of peak detection to {} ({:.3f}% of data length)".format(wlen, rel_wlen*100.))

peak, peak_info = scsig.find_peaks(fft_abs, prominence=prominence_thres, wlen=wlen)
peak_overtones = [[] for i in range(len(peak))]

if np.size(peak) < 100:
    logger.info("found {} peaks in fft".format(np.size(peak)))
else:
    logger.error("found too many peaks in fft ({})".format(np.size(peak)))

# load power supply specification from file
file_path_dir = os.path.abspath(os.path.dirname(__file__))
json_path = os.path.join(file_path_dir, "supplies.json")
with open(json_path) as json_file:
    supplies = json.load(json_file)
supplies_matches = [[] for i in range(len(supplies))]

logger.info("loaded specification of {} power supplies from file".format(len(supplies)))

# for each peak check if higher frequency peaks are integer multiples
for i in range(np.size(peak) - 1):
    for j in range(i + 1, np.size(peak)):
        if is_overtone(f[peak[i]], f[peak[j]], 1.0e-3):
           peak_overtones[j].append(i)

# check if peaks are close to the switching frequency of any specified supply
for i in range(np.size(peak)):
    match = False
    supplies_matches.append([])
    for j in range(len(supplies)):
        if isclose(f[peak[i]], supplies[j]["frequency"], supplies[j]["frequency_error"]):
            logger.info("matched peak at {:.3f} kHz with {} on {}".format(f[peak[i]]*1.e-3, supplies[j]["part"], supplies[j]["location"]))
            supplies_matches[j].append(i)
            match = True
    if len(peak_overtones[i]) > 0:
        for j in peak_overtones[i]:
            logger.info("matched peak at {:.3f} kHz as overtone of {:.3f} kHz (factor {:.3f})".format(f[peak[i]]*1.e-3, f[peak[j]]*1.e-3, f[peak[i]]/f[peak[j]]))
    else:
        if not match:
            logger.warn("peak at {:.3f} kHz not matched".format(f[peak[i]]*1.e-3))

############
# plottery #
############

if not os.path.exists("plots"):
    mkdir("plots")

plot_basename = "plots/w{}_h{}_a{}_".format(opts.wnr, opts.hnr, opts.anr)

# plot trace
plt_time = [1.e-3, 5.e-5, 10.e-6, 10.e-6, 2.e-6]
plt_range = ['start0', 'start0', 'start0', 'max', 'max']
for plt_idx_ in range(len(plt_time)):
    plt_range_idx = int(plt_time[plt_idx_] / d(t))
    if (plt_range[plt_idx_] == 'start0'):
        plt_stime_idx = 0
        plt_etime_idx = plt_range_idx
    elif (plt_range[plt_idx_] == 'max'):
        plt_max_idx = plt_range_idx + np.argmax(v[plt_range_idx:-plt_range_idx])
        plt_stime_idx = plt_max_idx - plt_range_idx//2
        plt_etime_idx = plt_max_idx + plt_range_idx//2
    plt.figure(figsize=(14,6))
    plt.axhline(y=v_mean, color="g")
    plt.axhline(y=v_mean + v_rms, color="r")
    plt.axhline(y=v_mean - v_rms, color="r")
    plt.plot(t[plt_stime_idx:plt_etime_idx]*1.e6, v[plt_stime_idx:plt_etime_idx] + v_mean, "b-")
    plt.xlabel("time [us]")
    plt.ylabel("amplitude [V]")
    plt.title("excerpt of recorded voltage trace ({})".format(plt_idx_))
    plt.tight_layout()
    plt.savefig(plot_basename+"trace_zoom_"+str(plt_idx_)+".png", dpi=300)
    plt.clf()

# plot overview of fft including detected peaks
plt.figure(figsize=(12,5))
plt.plot(f, dB(fft), "b-", alpha=.7)
plt.plot(f[peak], dB(fft[peak]), "ro")
plt.xscale("log")
plt.xlabel("frequency [Hz]")
plt.ylabel("fft amplitude [dB]")
plt.title("overview of {} detected peaks".format(np.size(peak)))
plt.tight_layout()
plt.savefig(plot_basename+"overview.png", dpi=300)
plt.clf()

# spectrogram
plt_time = 300.e-6
plt_time_idx = int(plt_time / d(t))
fs, ts, s = scsig.spectrogram(v[:plt_time_idx], 1./d(t), nperseg=int(np.size(v)/10000.), scaling='spectrum')
cmap = plt.get_cmap('gnuplot2')
plt.pcolormesh(ts * 1.e6, fs, s, cmap=cmap, shading='auto')
plt.ylabel("frequency [Hz]")
plt.xlabel("time [us]")
plt.title("spectrogram overview")
plt.tight_layout()
plt.savefig(plot_basename+"spectrogram_overview.png", dpi=300)
plt.clf()
cmap = plt.get_cmap('gist_heat')
plt.pcolormesh(ts * 1.e6, fs, dB(s)/2., cmap=cmap, shading='auto')
plt.ylabel("frequency [Hz]")
plt.xlabel("time [us]")
plt.title("spectrogram overview (log color scale)")
plt.tight_layout()
plt.savefig(plot_basename+"spectrogram_overview_log.png", dpi=300)
plt.clf()

plt_f = 2.e6
plt_time = 300.e-6
fs, ts, s = scsig.spectrogram(v, 1./d(t), nperseg=int(np.size(v)/500.), scaling='spectrum')
plt_f_idx = int(plt_f / d(fs))
plt_time_idx = int(plt_time / d(ts))
cmap = plt.get_cmap('gnuplot2')
plt.pcolormesh(ts * 1.e6, fs[:plt_f_idx], s[:plt_f_idx,:], cmap=cmap, shading='auto')
plt.ylabel("frequency [Hz]")
plt.xlabel("time [us]")
plt.title("spectrogram of low frequencies")
plt.tight_layout()
plt.savefig(plot_basename+"spectrogram_lowfreq.png", dpi=300)
plt.clf()
cmap = plt.get_cmap('gist_heat')
plt.pcolormesh(ts * 1.e6, fs[:plt_f_idx], dB(s[:plt_f_idx,:])/2., cmap=cmap, shading='auto')
plt.ylabel("frequency [Hz]")
plt.xlabel("time [us]")
plt.title("spectrogram of low frequencies (log color scale)")
plt.tight_layout()
plt.savefig(plot_basename+"spectrogram_lowfreq_log.png", dpi=300)
plt.clf()

# plot each supply switching frequency
supplies_id = [s["id"] for s in supplies]
for m in range(len(supplies)):
    plt.figure(figsize=(8,5))
    if len(supplies_matches[m]) > supplies[m]["count"]:
        logger.warn("too many matches ({} peaks vs {} database count) for {} on {}".format(len(supplies_matches[m]), supplies[m]["count"], supplies[m]["part"], supplies[m]["location"]))
    min_idx, max_idx = minmax_idx(supplies[m]["frequency"], supplies[m]["frequency_error"], d(f))
    plt.axvline(x=supplies[m]["frequency"], color="g")
    for x in supplies_matches[m]:
        plt.axvline(x=f[peak[x]], color="r")
    plt.plot(f[min_idx:max_idx], dB(fft[min_idx:max_idx]))
    if (len(supplies_matches[m]) > 0):
        plt.title("ID {}: {} on {}".format(supplies_id[m], supplies[m]["part"], supplies[m]["location"]))
    else:
        plt.title("ID {}: {} on {} (no matches found)".format(supplies_id[m], supplies[m]["part"], supplies[m]["location"]))
    plt.xlabel("frequency [Hz]")
    plt.ylabel("fft amplitude")
    plt.tight_layout()
    plt.savefig(plot_basename+"supply_{}.png".format(supplies_id[m]))
    plt.clf()


#############
# save data #
#############
np.save("anatrace_test_w{}_h{}_a{}".format(opts.wnr, opts.hnr, opts.anr), {"f":f_p, "fft_rect":fft_p, "fft_ft":fft, "peak_prominence_thres":prominence_thres, "peak_wlen":wlen, "peak":peak, "peak_info":peak_info, "peak_overtones":peak_overtones, "supplies_matches":supplies_matches, "supplies_id":supplies_id})
logger.info("saved data to file")

quit()
