#!/usr/bin/env python

import os
import copy


try:
    from waflib.extras import symwaf2ic
    from waflib.extras.gtest import summary
    recurse = lambda *args: None  # dummy recurse
except ImportError:
    from gtest import summary
    from symwaf2ic import repo, recurse_depends
    recurse = lambda ctx: recurse_depends(depends, ctx)


def depends(ctx):
    ctx('halbe', 'pyhalbe')
    ctx('pyhmf', 'pycellparameters')
    ctx('calibtic')


def options(opt):
    recurse(opt)


def configure(cfg):
    recurse(cfg)


def build(bld):
    recurse(bld)
