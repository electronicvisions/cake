#!/usr/bin/env python

#from waflib.extras import symwaf2ic
#from waflib.extras.gtest import summary
from waflib import Utils


def depends(ctx):
    ctx('sthal')
    ctx('logger', 'pylogging')
    ctx('calibtic', 'pycalibtic')  # sthal does not depend on pycalibtic
    ctx('redman', 'pyredman')
    # please specify 'waf setup --project=cake --with-sim' to
    # enable simulation, '--without-sim' to disable
    if ctx.options.with_sim:
        ctx('cd-denmem-teststand')

    ctx.recurse('test')


def options(opt):
    opt.load('doxygen')

    # add 'configure' options
    hopts = opt.add_option_group('Cake Options')
    hopts.add_withoption('sim', default=False,
                         help='Enable/Disable the simulation interface')

    opt.recurse('test')


def configure(cfg):
    cfg.load('doxygen')
    cfg.env.post_sim = cfg.options.with_sim

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
                 'redman_xml']
    if bld.env.post_sim:
        post_task.append('sim_denmem_ts')
    bld(
        target='pycake',
        source=bld.path.ant_glob('pycake/**/*.py'),
        features='py',
        depends_on = post_task,
        pythonpath=['.'],
        install_from='.',
        install_path='${PREFIX}/lib',
    )

    bld.install_files(
        '${PREFIX}/bin',
        bld.path.ant_glob('pycake/bin/cake_*') +
        bld.path.ant_glob('tools/cake_*') +
        bld.path.ant_glob('tools/digital_blacklisting/cake_blacklist_fpga.py') +
        bld.path.ant_glob('tools/L1/*'),
        relative_trick=False,
        chmod=Utils.O755,
    )

    bld.install_files(
        '${PREFIX}/bin/tools',
        bld.path.ant_glob('config/*py') +
        ['pycake/bin/overview.html'],
        chmod=Utils.O755,
        relative_trick=False
    )

    bld(
        target       = 'cake_digital_blacklisting',
        features     = 'cxx cxxprogram',
        source       = bld.path.ant_glob('tools/digital_blacklisting/digital_blacklisting.cpp'),
        use          = [ 'halco', 'halbe', 'redman', 'BOOST4TOOLS', 'hwdb4cpp', 'redman_xml'],
        install_path = '${PREFIX}/bin',
    )

    bld.recurse('test')


def doc(dcx):
    '''build documentation (doxygen)'''

    dcx(
        features='doxygen',
        doxyfile='doc/doxyfile',
        pdffile='pycake.pdf',
    )
