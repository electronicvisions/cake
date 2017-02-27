# How to start a cake pipeline with slurm-pipeline

### The pipeline
1. prerequisites (create folders, determine the current number of the run),
2. calibration (calibration measurements and fits),
3. plotting (plot calibration results),
4. evaluation (evaluation measurements),
5. plotting (plot evaluation results).
For short prereq-calib-plot-eval-plot.

The following lines explain how to set up your software and start a pipelined script.

### Setup

This part of the software uses [slurm-pipeline](https://github.com/acorg/slurm-pipeline). The
class is available in the visionary-wafer app, but it can be installed in other containers using
[spack package installation](https://brainscales-r.kip.uni-heidelberg.de/projects/symap2ic/wiki/Spack).

This tool requires loading modules `slurm-singularity` AND either module `nmpm_software` or
locally built software (localdir). In addition, variables CONTAINER_APP_NMPM_SOFTWARE and
CONTAINER_IMAGE_NMPM_SOFTWARE should be set (e.g. to `visionary-wafer` and `/containers/stable/latest`,
respectively). STHAL_DEFECTS_PATH can be set if blacklisting data should be inquired (e.g. in the case
that there are HICANNs without high-speed communication).

Currently, this tool copies the contents from the parameter folder supplied to slurm-pipeline.py
(p|param-folder) into the output folder (o|outdir) before starting its work. The calibration parameter
files should be present in this path (not as symbolic links).

If you use a local cake project, make sure that you have called
```
waf setup --project=cake --with-sim
waf -v configure
waf install -v -j8 --test-execnone --target=*
```
in your project's root directory, such that it is available for the pipeline scripts.

##### Example
```
slurm-pipeline.py --specification ./specification.json \
--output post.json --scriptArgs w=33 hicann=367 a=0 \
p=/path/to/params/ o=`readlink -f ./test`
```

All arguments for the pipeline are parsed in `slurm-pipeline/parse_input.sh`. Observe that
slurm-pipeline arguments are given with space separation, while arguments for the pipeline
scripts are given following `--scriptArgs` option with an equal sign (e.g. `--specification
specification.json` and `--scriptArgs w=33 a=0`). --scriptArgs does not accept parameters with
hyphen, e.g. -w=30 is not allowed. Alternatively, the syntax `--scriptArgs parameter1=<value>
--scriptArgs parameter2=<value>` is also possible.

Arguments:
* `--specification`: JSON file that contains the steps, with names, dependencies, etc.
* `--output`: JSON file that contains the output of
        slurm-pipeline.py (use `slurm-pipeline-status.py -s post.json` to view)
* `--scriptArgs`: Specify arguments that should appear on the command line when initial step
scripts are run. The provided arguments will be available to all called scripts through the
SP_ORIGINAL_ARGS variable
* Arguments for `--scriptArgs`:
*    `w|wafer` (int): wafer enum
*    `hicann` (int): HICANN enum
*    `a|aout` (int): analog out enum (0 or 1)
*    `p|param-folder` (str): a folder in which all parameters (`v4_params.py` etc. are)
*    `o|outdir` (str): path where the data will be saved (use `readlink -f ../relative/path`
     like in the example to convert a relative path to an absolute path)
*    `b|backend` (str): path to existing calibration files. This usage mode is for evaluation
     of existing calibrations. The existing calibration files are copied to `outdir/backends`
     before being loaded.
*    `new`: Start a new calibration, even if a folder already exists in this directory
*    `redo_fits`: Redo the fits, if the calibration has already completed

##### multiple logs are created during the process:
  * `submit_run${resume_number}.log`: output of the submit scripts
  * `slurm_${name}_run${resume_number}.log`: output of SLURM (SLURM errors are found here)
  * `cake_${calib, eval, post}_run${resume_number}.log`: output of cake scripts

variables:
  * name: name of the step, concatenated with previous steps (e.g., calib-prereq)
  * `resume_number`: tells you how often you reran the script (same wafer,
                HICANN AND workspace) => same resume_number in log files <=> same run

### How to start a pipeline for multiple HICANNs

Use `cake_calib_hicanns.sbatch`.

Arguments:
* `-w|--wafer` (int): the wafer enum
* `--hicanns` (int): HICANN(s) to calibrate ( -1 starts for all HICANNs, multiple
     HICANNs can be given like 1,65,2-5,3,90,100-149
* `-p|--param-folder` (str): directory where all parameter files (`v4_params.py` etc.) are in
* `-o|--outdir` (str): directory to save the calibrations to. A new folder is created
     in this directory for each HICANN
* `-b|--backend` (str): for evaluation of existing calibrations (view above).
* `--firstStep` (str, optional): start pipeline at this step
* `--lastStep` (str, optional): end pipeline at this step
* `--fpgas_to_exclude` (comma-sep int): Automatically determine which HICANNs
     Lay on these FPGAs and do not start a calibration for these even if they were given via
     `--hicanns`. This is useful e.g., if you pass `--hicanns -1` and then want to exclude some
     FPGAs that are not powered.
* `--switch_aouts` (flag, optional): If given, map even HICANNs to aout 1, else to aout 0.
* `--new` (flag, optional): run the pipeline even if there are already completed calibrations in
     outdir
* `--redo_fits` (flag, optional): ask cake_resume to redo fits. Default is to skip already
     performed fits.

##### Example:
```
cake_calib_hicanns.sbatch -w 21 --hicanns -1 -p /path/to/v4_params_etc/ -o \
`readlink -f 20190627_full_wafer`
```
The 'usual' workflow to calibrate a whole wafer can be to call the above command, wait for all
jobs to finish, plot the results using `cake_plot_wafer_status.py`.
If HICANNs are missing, reissue the above command and see if more calibrations are finished after
all jobs have finished.
To start different calibs of the same HICANN on the same wafer (e.g., to record `I_gl` with
`smallcap` and `bigcap`), use the `new` command for HICANNs that have already finished.
To test different fit functions etc. on data that you have already recorded, only add `redo_fits`
and call only finished HICANNs.
