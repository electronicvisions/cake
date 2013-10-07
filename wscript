#!/usr/bin/env python

import os
import copy

from waflib.extras import symwaf2ic
from waflib.extras.gtest import summary


def depends(ctx):
    ctx('symap2ic', 'src/pylogging')
    ctx('calibtic', 'pycalibtic')  # sthal does not depend on pycalibtic
    ctx('sthal')
    ctx('redman')

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
        post_task=['pyoneer', 'pycalibtic', '_pysthal', 'pylogging'],
        install_from='.',
        install_path='lib',
    )
