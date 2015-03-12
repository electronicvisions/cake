#!/usr/bin/env python

#from waflib.extras import symwaf2ic
#from waflib.extras.gtest import summary
from waflib import Utils


def depends(ctx):
    ctx('sthal')
    ctx('symap2ic', 'src/pylogging')
    ctx('calibtic', 'pycalibtic')  # sthal does not depend on pycalibtic
    ctx('redman', 'pyredman')
    ctx('redman', 'backends')
    ctx('cd-denmem-teststand')


def options(opt):
    opt.load('post_task')

    opt.recurse('test')


def configure(cfg):
    cfg.load('post_task')

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
    bld(
        target='pycake',
        source=bld.path.ant_glob('pycake/**/*.py'),
        features='post_task',
        post_task=['pycalibtic', 'pysthal', 'pyhalbe', 'pylogging', 'pyredman',
                   'redman_xml', 'redman_mongo', 'sim_denmem_ts', ],
        pythonpath=['.'],
        install_from='.',
        install_path='${PREFIX}/lib',
    )

    tools = ['pycake/bin/run_calibration.py', 'pycake/bin/resume.py', 'tools/run_test_calib.sh']
    bld.install_files(
        '${PREFIX}/bin/tools',
        tools,
        chmod=Utils.O755,
        relative_trick=False
    )
    bld.recurse('test')
