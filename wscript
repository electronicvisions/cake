#!/usr/bin/env python

import os
import copy

from waflib.extras import symwaf2ic
from waflib.extras.gtest import summary


def depends(ctx):
    ctx('sthal', branch='dev_polymorphic_handle')
    ctx('symap2ic', 'src/pylogging')
    ctx('calibtic', 'pycalibtic', branch='dev_sthal')  # sthal does not depend on pycalibtic
    ctx('redman', 'pyredman')
    ctx('redman', 'backends')

    if ctx.options.disable_bindings:
        ctx.fatal('Calibration depends on Python bindings.')


def options(opt):
    opt.load('post_task')


def configure(cfg):
    cfg.load('post_task')


def build(bld):
    bld(
        target='pycairo',
        source=bld.path.ant_glob('pycairo/**/*.py'),
        features='post_task',
        post_task=['pyoneer', 'pycalibtic', '_pysthal', 'pylogging', 'pyredman', 'redman_xml', 'redman_mongo'],
        install_from='.',
        install_path='lib',
    )
