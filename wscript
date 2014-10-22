#!/usr/bin/env python

#from waflib.extras import symwaf2ic
#from waflib.extras.gtest import summary


def depends(ctx):
    ctx('sthal')
    ctx('symap2ic', 'src/pylogging')
    ctx('calibtic', 'pycalibtic')  # sthal does not depend on pycalibtic
    ctx('redman', 'pyredman')
    ctx('redman', 'backends')


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
                   'redman_xml', 'redman_mongo'],
        pythonpath=['.'],
        install_from='.',
        install_path='lib',
    )
    bld.recurse('test')
