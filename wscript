#!/usr/bin/env python

#from waflib.extras import symwaf2ic
#from waflib.extras.gtest import summary
from waflib import Utils


def depends(ctx):
    ctx('sthal')
    ctx('logger', 'pylogging')
    ctx('calibtic', 'pycalibtic')  # sthal does not depend on pycalibtic
    ctx('redman', 'pyredman')
    ctx('redman', 'backends')
    # please specify 'waf setup --project=cake --with-sim' to
    # enable simulation, '--without-sim' to disable
    if ctx.options.with_sim:
        ctx('cd-denmem-teststand')


def options(opt):
    opt.load('post_task')
    opt.load('documentation')

    # add 'configure' options
    hopts = opt.add_option_group('Cake Options')
    hopts.add_withoption('sim', default=False,
                         help='Enable/Disable the simulation interface')

    opt.recurse('test')


def configure(cfg):
    cfg.load('post_task')
    cfg.load('documentation')
    cfg.env.post_sim = cfg.options.with_sim
    # needed for doc build
    cfg.find_program("doxypy", mandatory=True)

    try:
        disable_bindings = cfg.options.disable_bindings
    except AttributeError:
        try:
            disable_bindings = not cfg.options.bindings
        except AttributeError:
            disable_bindings = not cfg.env.build_python_bindings

    if disable_bindings:
        cfg.fatal('Calibration depends on Python bindings.')

    cfg.recurse('test')


def build(bld):
    post_task = ['pycalibtic', 'pysthal', 'pyhalbe', 'pylogging', 'pyredman',
                 'redman_xml', 'redman_mongo']
    if bld.env.post_sim:
        post_task.append('sim_denmem_ts')
    bld(
        target='pycake',
        source=bld.path.ant_glob('pycake/**/*.py'),
        features='py post_task',
        post_task=post_task,
        pythonpath=['.'],
        install_from='.',
        install_path='${PREFIX}/lib',
    )

    tools = [
        'pycake/bin/run_calibration.py',
        'pycake/bin/resume.py',
        'pycake/bin/make_plots.py',
        'pycake/bin/overview.html',
        'tools/run_test_calib.sh',
        'pycake/bin/plot_calib_run',
    ] + bld.path.ant_glob('config/*py')
    bld.install_files(
        '${PREFIX}/bin/tools',
        tools,
        chmod=Utils.O755,
        relative_trick=False
    )
    bld.recurse('test')


def doc(dcx):
    '''build documentation (doxygen)'''

    dcx(
        features='doxygen',
        doxyfile='doc/doxyfile',
        pdffile='pycake.pdf',
    )
