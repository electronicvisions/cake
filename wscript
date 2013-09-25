#!/usr/bin/env python

import os
import copy

from waflib.extras import symwaf2ic
from waflib.extras.gtest import summary


def depends(ctx):
    #ctx('halbe', 'pyhalbe')
    #ctx('pyhmf', 'pycellparameters')
    ctx('calibtic', 'pycalibtic')  # sthal does not depend on pycalibtic
    ctx('sthal')
    ctx('redman')

    if ctx.options.disable_bindings:
        ctx.fatal('Calibration depends on Python bindings.')


def options(opt):
    pass


def configure(cfg):
    pass


def build(bld):
    bld.install_files(
            'lib',
            bld.path.ant_glob('pycairo/**/*.py'),
            relative_trick=True,
    )
